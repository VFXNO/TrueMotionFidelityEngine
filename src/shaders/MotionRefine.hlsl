// ============================================================================
// PROFESSIONAL MOTION REFINEMENT - Full Resolution
// Sub-pixel accurate refinement with artifact prevention
// Author: Professional Shader Engineer
// ============================================================================

Texture2D<float> PrevLuma : register(t0);
Texture2D<float> CurrLuma : register(t1);
Texture2D<float2> CoarseMotion : register(t2);
Texture2D<float> CoarseConf : register(t3);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfidenceOut : register(u1);

SamplerState LinearClamp : register(s0);

cbuffer RefineCB : register(b0) {
    int radius;
    float motionScale;
    float2 pad;
};

// ============================================================================
// CONSTANTS - Tuned for smooth, artifact-free motion
// ============================================================================
static const float SUBPIXEL_STEP = 0.5;
static const float CONF_SCALE = 4.5;
static const float SMALL_OBJECT_WEIGHT = 3.0;      // Center pixel importance
static const float SMOOTHNESS_PENALTY = 0.0015;    // Deviation from coarse penalty
static const float NEIGHBOR_CONSISTENCY_WEIGHT = 0.4;  // Weight for neighbor agreement

// Gaussian weights for 5x5 patch (sigma=1.2) - wider for smoother matching
static const float kGaussian5x5[25] = {
    0.0183, 0.0335, 0.0415, 0.0335, 0.0183,
    0.0335, 0.0613, 0.0760, 0.0613, 0.0335,
    0.0415, 0.0760, 0.0943, 0.0760, 0.0415,
    0.0335, 0.0613, 0.0760, 0.0613, 0.0335,
    0.0183, 0.0335, 0.0415, 0.0335, 0.0183
};

// ============================================================================
// HELPER: Bilinear luma sampling
// ============================================================================
float SampleLuma(Texture2D<float> tex, float2 pos, float2 size) {
    float2 clamped = clamp(pos, float2(0.5, 0.5), size - float2(0.5, 0.5));
    float2 uv = clamped / size;
    return tex.SampleLevel(LinearClamp, uv, 0);
}

// ============================================================================
// HELPER: Enhanced patch cost with adaptive weighting
// ============================================================================
float ComputePatchCost(int2 currBase, float2 offset, float centerLuma,
                       uint width, uint height, float localVariance)
{
    float2 size = float2(width, height);
    
    // Sample center for DC offset (zero-mean matching)
    float2 prevCenterPos = float2(currBase) + offset;
    float prevCenterLuma = SampleLuma(PrevLuma, prevCenterPos, size);
    float dcOffset = prevCenterLuma - centerLuma;
    
    float cost = 0.0;
    float weightSum = 0.0;
    
    // Adaptive patch size based on local variance
    // Low variance (flat) = larger effective patch for stability
    // High variance (texture) = smaller effective patch for accuracy
    float patchSizeModifier = lerp(1.2, 0.9, saturate(localVariance * 10.0));
    
    // 5x5 patch matching with Gaussian weighting
    [unroll]
    for (int dy = -2; dy <= 2; ++dy) {
        [unroll]
        for (int dx = -2; dx <= 2; ++dx) {
            int idx = (dy + 2) * 5 + (dx + 2);
            float baseWeight = kGaussian5x5[idx];
            
            // Extra weight for center region (helps small objects)
            float centerBoost = (abs(dx) <= 1 && abs(dy) <= 1) ? SMALL_OBJECT_WEIGHT : 1.0;
            float weight = baseWeight * centerBoost * patchSizeModifier;
            
            // Sample current and previous
            int2 currPos = clamp(currBase + int2(dx, dy), int2(0, 0), int2(width - 1, height - 1));
            float currVal = CurrLuma.Load(int3(currPos, 0));
            
            float2 prevPos = float2(currBase) + offset + float2(dx, dy);
            float prevVal = SampleLuma(PrevLuma, prevPos, size);
            
            // Zero-mean SAD with SSD component for robustness
            float diff = (prevVal - currVal) - dcOffset;
            float sad = abs(diff);
            float ssd = diff * diff;
            
            // Combine SAD and SSD (SSD penalizes large errors more)
            cost += (sad * 0.6 + sqrt(ssd) * 0.4) * weight;
            weightSum += weight;
        }
    }
    
    return cost / max(weightSum, 0.001);
}

// ============================================================================
// HELPER: Compute local variance for adaptive processing
// ============================================================================
float ComputeLocalVariance(int2 base, float centerLuma, uint width, uint height)
{
    float variance = 0.0;
    
    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            int2 pos = clamp(base + int2(dx, dy), int2(0, 0), int2(width - 1, height - 1));
            float val = CurrLuma.Load(int3(pos, 0));
            variance += abs(val - centerLuma);
        }
    }
    
    return variance / 9.0;
}

// ============================================================================
// HELPER: Sub-pixel refinement using parabola fitting
// ============================================================================
float2 SubPixelRefine(int2 base, float2 intOffset, uint width, uint height, 
                      float centerLuma, float localVariance)
{
    // Sample costs at half-pixel offsets
    float costC = ComputePatchCost(base, intOffset, centerLuma, width, height, localVariance);
    float costL = ComputePatchCost(base, intOffset + float2(-SUBPIXEL_STEP, 0), centerLuma, width, height, localVariance);
    float costR = ComputePatchCost(base, intOffset + float2(SUBPIXEL_STEP, 0), centerLuma, width, height, localVariance);
    float costU = ComputePatchCost(base, intOffset + float2(0, -SUBPIXEL_STEP), centerLuma, width, height, localVariance);
    float costD = ComputePatchCost(base, intOffset + float2(0, SUBPIXEL_STEP), centerLuma, width, height, localVariance);
    
    // Parabola fitting
    float denomX = costL + costR - 2.0 * costC;
    float denomY = costU + costD - 2.0 * costC;
    
    float subX = (abs(denomX) > 1e-5) ? (SUBPIXEL_STEP * (costL - costR) / denomX) : 0.0;
    float subY = (abs(denomY) > 1e-5) ? (SUBPIXEL_STEP * (costU - costD) / denomY) : 0.0;
    
    // Clamp sub-pixel offset and apply confidence-based damping
    // Larger sub-pixel corrections are less reliable
    float subMag = length(float2(subX, subY));
    float damping = saturate(1.0 - subMag * 0.5);
    
    subX = clamp(subX * damping, -SUBPIXEL_STEP, SUBPIXEL_STEP);
    subY = clamp(subY * damping, -SUBPIXEL_STEP, SUBPIXEL_STEP);
    
    return intOffset + float2(subX, subY);
}

// ============================================================================
// HELPER: Check neighbor motion consistency
// ============================================================================
float ComputeNeighborConsistency(float2 motion, float2 uv, float2 invSize)
{
    float2 neighbors[4];
    neighbors[0] = CoarseMotion.SampleLevel(LinearClamp, uv + float2(-1, 0) * invSize, 0);
    neighbors[1] = CoarseMotion.SampleLevel(LinearClamp, uv + float2(1, 0) * invSize, 0);
    neighbors[2] = CoarseMotion.SampleLevel(LinearClamp, uv + float2(0, -1) * invSize, 0);
    neighbors[3] = CoarseMotion.SampleLevel(LinearClamp, uv + float2(0, 1) * invSize, 0);
    
    float consistency = 0.0;
    float motionMag = length(motion);
    
    [unroll]
    for (int i = 0; i < 4; ++i) {
        float nMag = length(neighbors[i]);
        float magSim = 1.0 - abs(motionMag - nMag) / (max(motionMag, nMag) + 0.5);
        
        float dirSim = 1.0;
        if (motionMag > 0.3 && nMag > 0.3) {
            dirSim = dot(normalize(motion), normalize(neighbors[i])) * 0.5 + 0.5;
        }
        
        consistency += magSim * dirSim * 0.25;
    }
    
    return consistency;
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    CurrLuma.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height) return;
    
    int2 base = int2(id.xy);
    float centerLuma = CurrLuma.Load(int3(base, 0));
    
    // Compute local variance for adaptive processing
    float localVariance = ComputeLocalVariance(base, centerLuma, width, height);
    
    // ========================================================================
    // STEP 1: Get coarse motion prediction (upscaled with bilinear interpolation)
    // ========================================================================
    uint cw, ch;
    CoarseMotion.GetDimensions(cw, ch);
    
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    float2 invSize = 1.0 / float2(width, height);
    float2 coarse = CoarseMotion.SampleLevel(LinearClamp, uv, 0);
    float coarseConf = CoarseConf.SampleLevel(LinearClamp, uv, 0);
    
    // Scale coarse motion to full resolution
    float2 pred = coarse * motionScale;
    int2 baseOffset = int2(round(pred));
    
    // ========================================================================
    // STEP 2: Adaptive search radius based on coarse confidence
    // Low confidence = wider search, high confidence = narrow search
    // ========================================================================
    int searchRadius = (int)lerp(float(max(radius, 2)), float(max((uint)radius / 2u, 1u)), coarseConf);
    searchRadius = clamp(searchRadius, 1, 5);
    
    // ========================================================================
    // STEP 3: Integer-pixel search around prediction
    // ========================================================================
    float bestCost = 1e9;
    int2 bestOffset = baseOffset;
    
    for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
        for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
            int2 offset = baseOffset + int2(dx, dy);
            float cost = ComputePatchCost(base, float2(offset), centerLuma, width, height, localVariance);
            
            // Distance penalty - prefer staying close to coarse prediction for smoothness
            float dist = length(float2(dx, dy));
            cost += dist * SMOOTHNESS_PENALTY;
            
            // Higher penalty for low-confidence coarse predictions
            cost += dist * (1.0 - coarseConf) * SMOOTHNESS_PENALTY * 2.0;
            
            if (cost < bestCost) {
                bestCost = cost;
                bestOffset = offset;
            }
        }
    }
    
    // ========================================================================
    // STEP 4: Sub-pixel refinement
    // ========================================================================
    float2 refined = SubPixelRefine(base, float2(bestOffset), width, height, centerLuma, localVariance);
    
    // ========================================================================
    // STEP 5: Compute confidence with multiple factors
    // ========================================================================
    float finalCost = ComputePatchCost(base, refined, centerLuma, width, height, localVariance);
    float matchConf = exp(-finalCost * CONF_SCALE);
    
    // Check neighbor consistency for artifact suppression
    float neighborConsistency = ComputeNeighborConsistency(refined, uv, invSize);
    
    // Combine confidence factors
    float confidence = matchConf;
    
    // Reduce confidence if motion disagrees with neighbors (potential artifact)
    confidence *= lerp(0.5, 1.0, neighborConsistency);
    
    // Blend with coarse confidence (provides temporal stability)
    float coarseWeight = saturate(coarseConf * 0.6 + 0.4);
    confidence *= coarseWeight;
    
    // Low-texture regions need lower confidence (less reliable matching)
    float textureConf = saturate(localVariance * 15.0 + 0.3);
    confidence *= textureConf;
    
    confidence = saturate(confidence);
    
    // ========================================================================
    // STEP 6: Smooth blend with coarse motion for stability
    // If our refinement is uncertain, stay closer to coarse prediction
    // ========================================================================
    float blendWeight = saturate(confidence * neighborConsistency);
    float2 finalMotion = lerp(pred, refined, blendWeight * 0.7 + 0.3);
    
    // ========================================================================
    // OUTPUT
    // ========================================================================
    MotionOut[id.xy] = finalMotion;
    ConfidenceOut[id.xy] = confidence;
}
