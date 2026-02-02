// ============================================================================
// PROFESSIONAL MOTION ESTIMATION - Coarse Pass (Quarter Resolution)
// Hierarchical block matching with multi-predictor consensus
// Author: Professional Shader Engineer
// ============================================================================

Texture2D<float> PrevLuma : register(t0);
Texture2D<float> CurrLuma : register(t1);
Texture2D<float2> MotionPred : register(t2);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfidenceOut : register(u1);

cbuffer MotionCB : register(b0) {
    int radius;
    int usePrediction;
    float2 pad;
};

// ============================================================================
// TUNING CONSTANTS - Carefully balanced for gaming content
// ============================================================================

// Cost function weights
static const float WEIGHT_SAD = 0.45;          // Sum of Absolute Differences
static const float WEIGHT_SSD = 0.35;          // Sum of Squared Differences  
static const float WEIGHT_CENSUS = 0.20;       // Census transform (illumination invariant)

// Region classification thresholds
static const float VARIANCE_FLAT = 0.012;      // Below = flat region (aperture problem)
static const float VARIANCE_TEXTURE = 0.045;   // Above = textured region
static const float GRADIENT_EDGE = 0.025;      // Above = edge region

// Smoothness regularization
static const float LAMBDA_SPATIAL = 0.0025;    // Smoothness cost weight
static const float LAMBDA_TEMPORAL = 0.0018;   // Temporal consistency weight

// Confidence model
static const float CONF_SCALE = 8.0;           // Cost to confidence mapping
static const float CONF_MIN = 0.12;            // Minimum confidence floor
static const float CONF_MAX = 0.96;            // Maximum confidence cap

// ============================================================================
// GAUSSIAN WEIGHTS (5x5 precomputed, sigma=1.5)
// ============================================================================
static const float kGaussian5x5[25] = {
    0.0232, 0.0338, 0.0383, 0.0338, 0.0232,
    0.0338, 0.0492, 0.0558, 0.0492, 0.0338,
    0.0383, 0.0558, 0.0632, 0.0558, 0.0383,
    0.0338, 0.0492, 0.0558, 0.0492, 0.0338,
    0.0232, 0.0338, 0.0383, 0.0338, 0.0232
};

// ============================================================================
// HELPER: Robust Multi-Metric Cost Function
// Combines SAD, SSD, and Census for robustness across different scenarios
// ============================================================================
float ComputeBlockCost(int2 currBase, int2 offset, float centerLuma,
                       float cachedBlock[25], uint width, uint height)
{
    // Sample center of previous frame at offset position
    int2 prevCenter = clamp(currBase + offset, int2(0, 0), int2(width - 1, height - 1));
    float prevCenterLuma = PrevLuma.Load(int3(prevCenter, 0));
    
    // DC offset for zero-mean matching (handles brightness changes)
    float dcOffset = prevCenterLuma - centerLuma;
    
    float sadSum = 0.0;
    float ssdSum = 0.0;
    float censusSum = 0.0;
    float weightSum = 0.0;
    
    // 5x5 block matching with Gaussian weighting
    [unroll]
    for (int dy = -2; dy <= 2; ++dy) {
        [unroll]
        for (int dx = -2; dx <= 2; ++dx) {
            int idx = (dy + 2) * 5 + (dx + 2);
            float weight = kGaussian5x5[idx];
            
            // Current frame value from cache
            float currVal = cachedBlock[idx];
            
            // Previous frame value with offset
            int2 prevPos = clamp(currBase + offset + int2(dx, dy), int2(0, 0), int2(width - 1, height - 1));
            float prevVal = PrevLuma.Load(int3(prevPos, 0));
            
            // Zero-mean difference (invariant to global brightness)
            float diff = (prevVal - currVal) - dcOffset;
            
            // SAD component
            sadSum += abs(diff) * weight;
            
            // SSD component (more sensitive to large errors)
            ssdSum += diff * diff * weight;
            
            // Census component (sign comparison - illumination invariant)
            float prevSign = (prevVal > prevCenterLuma) ? 1.0 : 0.0;
            float currSign = (currVal > centerLuma) ? 1.0 : 0.0;
            censusSum += abs(prevSign - currSign) * weight;
            
            weightSum += weight;
        }
    }
    
    // Normalize
    float invWeight = 1.0 / max(weightSum, 0.001);
    sadSum *= invWeight;
    ssdSum *= invWeight;
    censusSum *= invWeight;
    
    // Combine metrics (SSD is already squared, take sqrt for consistency)
    return sadSum * WEIGHT_SAD + sqrt(ssdSum) * WEIGHT_SSD + censusSum * WEIGHT_CENSUS;
}

// ============================================================================
// HELPER: Compute Local Variance and Gradient
// ============================================================================
void AnalyzeRegion(float cachedBlock[25], float centerLuma,
                   out float variance, out float2 gradient, out float edgeness)
{
    // Variance using center-weighted samples
    float sumDiff = 0.0;
    float sumWeight = 0.0;
    
    [unroll]
    for (int i = 0; i < 25; ++i) {
        float w = kGaussian5x5[i];
        sumDiff += abs(cachedBlock[i] - centerLuma) * w;
        sumWeight += w;
    }
    variance = sumDiff / max(sumWeight, 0.001);
    
    // Sobel gradient (3x3 center of the 5x5 block)
    float gx = -cachedBlock[6] + cachedBlock[8]
             - 2.0 * cachedBlock[11] + 2.0 * cachedBlock[13]
             - cachedBlock[16] + cachedBlock[18];
    
    float gy = -cachedBlock[6] - 2.0 * cachedBlock[7] - cachedBlock[8]
             + cachedBlock[16] + 2.0 * cachedBlock[17] + cachedBlock[18];
    
    gradient = float2(gx, gy) / 8.0;
    edgeness = length(gradient);
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
    int maxMag = int(min(width, height) / 3u);
    
    // ========================================================================
    // STEP 1: Cache 5x5 block from current frame
    // ========================================================================
    float cachedBlock[25];
    [unroll]
    for (int cy = -2; cy <= 2; ++cy) {
        [unroll]
        for (int cx = -2; cx <= 2; ++cx) {
            int2 p = clamp(base + int2(cx, cy), int2(0, 0), int2(width - 1, height - 1));
            cachedBlock[(cy + 2) * 5 + (cx + 2)] = CurrLuma.Load(int3(p, 0));
        }
    }
    
    // ========================================================================
    // STEP 2: Analyze local region characteristics
    // ========================================================================
    float variance, edgeness;
    float2 gradient;
    AnalyzeRegion(cachedBlock, centerLuma, variance, gradient, edgeness);
    
    float flatness = saturate(1.0 - variance / VARIANCE_FLAT);
    float textureness = saturate((variance - VARIANCE_FLAT) / (VARIANCE_TEXTURE - VARIANCE_FLAT));
    
    // ========================================================================
    // STEP 3: Detect static regions
    // ========================================================================
    float prevCenterLuma = PrevLuma.Load(int3(base, 0));
    float frameDiff = abs(centerLuma - prevCenterLuma);
    
    float maxLocalDiff = frameDiff;
    [unroll]
    for (int sy = -1; sy <= 1; ++sy) {
        [unroll]
        for (int sx = -1; sx <= 1; ++sx) {
            int2 sPos = clamp(base + int2(sx, sy), int2(0, 0), int2(width - 1, height - 1));
            float c = CurrLuma.Load(int3(sPos, 0));
            float p = PrevLuma.Load(int3(sPos, 0));
            maxLocalDiff = max(maxLocalDiff, abs(c - p));
        }
    }
    
    // Static detection
    bool isStatic = (maxLocalDiff < 0.018) && (flatness > 0.5 || variance < 0.02);
    if (isStatic) {
        MotionOut[id.xy] = float2(0, 0);
        ConfidenceOut[id.xy] = 0.92;
        return;
    }
    
    // ========================================================================
    // STEP 4: Initialize search
    // ========================================================================
    float bestCost = 1e9;
    int2 bestOffset = int2(0, 0);
    
    // ========================================================================
    // STEP 5: Evaluate candidates
    // ========================================================================
    
    #define EVAL_CANDIDATE(offset, bonus) { \
        float cost = ComputeBlockCost(base, offset, centerLuma, cachedBlock, width, height); \
        cost *= bonus; \
        if (cost < bestCost) { bestCost = cost; bestOffset = offset; } \
    }
    
    // Zero motion
    float staticBonus = lerp(0.95, 0.80, flatness); // Stronger bias for flat regions
    if (isStatic) staticBonus *= 0.5; // Strong preference for static if detected
    EVAL_CANDIDATE(int2(0, 0), staticBonus);
    
    // TEMPORAL PREDICTION (SAFE)
    if (usePrediction) {
        float2 pred = MotionPred.Load(int3(base, 0));
        int2 prevVec = int2(pred);
        prevVec = clamp(prevVec, int2(-maxMag, -maxMag), int2(maxMag, maxMag));
        
        // Only test if it's significant
        if (any(prevVec != int2(0, 0))) {
            // Apply slight bonus to encourage temporal coherence, but not blindly
            // 0.96 bonus means it must be almost as good as the zero vector
            EVAL_CANDIDATE(prevVec, 0.96);
        }
    }

    // ========================================================================
    // STEP 6: Local search refinement
    // ========================================================================
    int searchRadius = clamp(radius, 2, 8);
    int2 searchCenter = bestOffset;
    int step = (searchRadius > 4) ? 2 : 1;
    
    // Regularization constant - higher = cleaner field
    float lambdaDist = 0.005; 

    for (int dy = -searchRadius; dy <= searchRadius; dy += step) {
        for (int dx = -searchRadius; dx <= searchRadius; dx += step) {
            int2 testOffset = searchCenter + int2(dx, dy);
            testOffset = clamp(testOffset, int2(-maxMag, -maxMag), int2(maxMag, maxMag));
            
            float cost = ComputeBlockCost(base, testOffset, centerLuma, cachedBlock, width, height);
            
            float dist = length(float2(dx, dy));
            // Apply distance penalty (regularization)
            cost += dist * lambdaDist;
            
            if (flatness > 0.5) {
                // In flat regions, penalize motion heavily to prevent noise tracking
                cost += dist * 0.03 * flatness;
            }
            
            if (cost < bestCost) {
                bestCost = cost;
                bestOffset = testOffset;
            }
        }
    }
    
    // Fine search
    if (step > 1) {
        searchCenter = bestOffset;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                
                int2 testOffset = searchCenter + int2(dx, dy);
                float cost = ComputeBlockCost(base, testOffset, centerLuma, cachedBlock, width, height);
                
                // Consistency: Apply same regularization in fine search
                float dist = length(float2(testOffset));
                cost += dist * lambdaDist;
                
                if (cost < bestCost) {
                    bestCost = cost;
                    bestOffset = testOffset;
                }
            }
        }
    }
    
    #undef EVAL_CANDIDATE
    
    // ========================================================================
    // STEP 8: Compute confidence
    // ========================================================================
    float matchConf = exp(-bestCost * CONF_SCALE);
    
    float regionMod = 1.0;
    regionMod *= lerp(1.0, 0.7, flatness);
    regionMod *= lerp(1.0, 0.85, textureness);
    regionMod *= lerp(1.0, 1.1, saturate(edgeness / GRADIENT_EDGE));
    
    float confidence = clamp(matchConf * regionMod, CONF_MIN, CONF_MAX);
    
    // ========================================================================
    // OUTPUT
    // ========================================================================
    MotionOut[id.xy] = float2(bestOffset);
    ConfidenceOut[id.xy] = confidence;
}
