// ============================================================================
// PROFESSIONAL MOTION SMOOTHING - Conservative, Sharp-Preserving
// Removes ONLY clear artifacts, preserves motion sharpness
// Author: Professional Shader Engineer
// ============================================================================

Texture2D<float2> MotionIn : register(t0);
Texture2D<float> ConfIn : register(t1);
Texture2D<float> LumaIn : register(t2);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfOut : register(u1);

cbuffer SmoothCB : register(b0) {
    float edgeScale;
    float confPower;
    float2 pad;
};

// ============================================================================
// CONSTANTS - Conservative settings to avoid blur
// ============================================================================
static const float OUTLIER_THRESHOLD = 2.5;     // Only fix clear outliers
static const float MIN_SMOOTHING = 0.05;        // Very minimal default smoothing
static const float MAX_SMOOTHING = 0.35;        // Cap on smoothing even for outliers
static const float EDGE_PRESERVE_STRENGTH = 0.8; // Strong edge preservation

// ============================================================================
// HELPER: Check if pixel is a clear outlier
// ============================================================================
struct OutlierInfo {
    bool isOutlier;
    float2 replacementMotion;
    float severity;
};

OutlierInfo DetectOutlier(int2 base, float2 centerMotion, float centerConf, 
                          uint width, uint height)
{
    OutlierInfo info;
    info.isOutlier = false;
    info.replacementMotion = centerMotion;
    info.severity = 0.0;
    
    float centerMag = length(centerMotion);
    
    // Sample 4 direct neighbors only (fast, robust)
    int2 offsets[4] = { int2(-1,0), int2(1,0), int2(0,-1), int2(0,1) };
    
    float2 neighborSum = float2(0, 0);
    float neighborMagSum = 0.0;
    int agreementCount = 0;
    int totalNeighbors = 0;
    
    [unroll]
    for (int i = 0; i < 4; ++i) {
        int2 nPos = clamp(base + offsets[i], int2(0,0), int2(width-1, height-1));
        float2 nMotion = MotionIn.Load(int3(nPos, 0));
        float nMag = length(nMotion);
        
        neighborSum += nMotion;
        neighborMagSum += nMag;
        totalNeighbors++;
        
        // Check if this neighbor agrees with center
        float magDiff = abs(centerMag - nMag);
        float dirSim = 1.0;
        if (centerMag > 0.5 && nMag > 0.5) {
            dirSim = dot(normalize(centerMotion), normalize(nMotion));
        }
        
        if (magDiff < 1.5 && dirSim > 0.5) {
            agreementCount++;
        }
    }
    
    float2 neighborMean = neighborSum / 4.0;
    float neighborMeanMag = neighborMagSum / 4.0;
    
    // Outlier detection: center disagrees with ALL neighbors
    if (agreementCount == 0) {
        float deviation = length(centerMotion - neighborMean);
        
        // Strong outlier: motion very different from neighbors
        if (deviation > OUTLIER_THRESHOLD) {
            info.isOutlier = true;
            info.replacementMotion = neighborMean;
            info.severity = saturate((deviation - OUTLIER_THRESHOLD) / 3.0);
        }
        // Isolated motion in static area
        else if (centerMag > 1.0 && neighborMeanMag < 0.3) {
            info.isOutlier = true;
            info.replacementMotion = float2(0, 0);
            info.severity = 0.8;
        }
    }
    
    return info;
}
// ============================================================================
// HELPER: Edge-aware weight (only smooth within same object)
// ============================================================================
float ComputeEdgeWeight(float centerLuma, float neighborLuma, float edgeScale)
{
    float lumaDiff = abs(centerLuma - neighborLuma) * edgeScale;
    // Sharp cutoff at edges
    return (lumaDiff < 0.1) ? 1.0 : exp(-lumaDiff * lumaDiff * 50.0);
}

// ============================================================================
// MAIN ENTRY POINT - Conservative, Sharp-Preserving
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    MotionIn.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height) return;
    
    int2 base = int2(id.xy);
    
    // Center pixel values
    float2 centerMotion = MotionIn.Load(int3(base, 0));
    float centerConf = ConfIn.Load(int3(base, 0));
    float centerLuma = LumaIn.Load(int3(base, 0));
    
    // ========================================================================
    // STEP 1: Outlier detection (fast, 4-neighbor check)
    // ========================================================================
    OutlierInfo outlier = DetectOutlier(base, centerMotion, centerConf, width, height);
    
    // ========================================================================
    // STEP 2: Decide action based on outlier status
    // ========================================================================
    float2 finalMotion;
    float finalConf;
    
    if (outlier.isOutlier && outlier.severity > 0.5) {
        // CLEAR OUTLIER: Replace with neighbor estimate
        float blend = saturate(outlier.severity * MAX_SMOOTHING * 2.0);
        finalMotion = lerp(centerMotion, outlier.replacementMotion, blend);
        finalConf = centerConf * (1.0 - outlier.severity * 0.3);
    }
    else if (outlier.isOutlier) {
        // MILD OUTLIER: Gentle blend
        float blend = outlier.severity * MAX_SMOOTHING;
        finalMotion = lerp(centerMotion, outlier.replacementMotion, blend);
        finalConf = centerConf;
    }
    else {
        // NOT AN OUTLIER: Apply very minimal edge-aware smoothing
        // This reduces temporal jitter without blurring motion
        
        int2 offsets[4] = { int2(-1,0), int2(1,0), int2(0,-1), int2(0,1) };
        float2 weightedSum = centerMotion * 4.0; // Strong center weight
        float totalWeight = 4.0;
        
        [unroll]
        for (int i = 0; i < 4; ++i) {
            int2 nPos = clamp(base + offsets[i], int2(0,0), int2(width-1, height-1));
            float2 nMotion = MotionIn.Load(int3(nPos, 0));
            float nLuma = LumaIn.Load(int3(nPos, 0));
            float nConf = ConfIn.Load(int3(nPos, 0));
            
            // Only blend if same object (edge-aware)
            float edgeW = ComputeEdgeWeight(centerLuma, nLuma, edgeScale);
            
            // Only blend if similar motion (motion-aware)
            float motionSim = 1.0 - saturate(length(nMotion - centerMotion) / 3.0);
            
            float weight = edgeW * motionSim * MIN_SMOOTHING;
            weightedSum += nMotion * weight;
            totalWeight += weight;
        }
        
        finalMotion = weightedSum / totalWeight;
        finalConf = centerConf;
    }
    
    // ========================================================================
    // STEP 3: Edge preservation (never smooth across strong luma edges)
    // ========================================================================
    float lumaGradient = 0.0;
    int2 cardinals[4] = { int2(-1,0), int2(1,0), int2(0,-1), int2(0,1) };
    [unroll]
    for (int j = 0; j < 4; ++j) {
        int2 nPos = clamp(base + cardinals[j], int2(0,0), int2(width-1, height-1));
        float nLuma = LumaIn.Load(int3(nPos, 0));
        lumaGradient = max(lumaGradient, abs(nLuma - centerLuma));
    }
    
    // At edges: prefer original motion
    if (lumaGradient > 0.06) {
        float edgePreserve = saturate((lumaGradient - 0.06) * EDGE_PRESERVE_STRENGTH * 15.0);
        finalMotion = lerp(finalMotion, centerMotion, edgePreserve);
    }
    
    // ========================================================================
    // OUTPUT - Pass through with minimal modification
    // ========================================================================
    MotionOut[id.xy] = finalMotion;
    ConfOut[id.xy] = saturate(finalConf);
}
