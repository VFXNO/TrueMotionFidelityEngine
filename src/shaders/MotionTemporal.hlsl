// ============================================================================
// MOTION TEMPORAL - Temporal Accumulation with Neighborhood Clamping
// Clean temporal filtering for stable motion estimation
// ============================================================================

Texture2D<float2> MotionCurr : register(t0);
Texture2D<float> ConfCurr : register(t1);
Texture2D<float2> MotionHistory : register(t2);
Texture2D<float> ConfHistory : register(t3);
Texture2D<float> LumaPrev : register(t4);
Texture2D<float> LumaCurr : register(t5);

RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfOut : register(u1);

SamplerState LinearClamp : register(s0);

cbuffer TemporalCB : register(b0) {
    float historyWeight;    // Base history blend (0.0-0.9)
    float confInfluence;    // How much confidence affects blend
    int resetHistory;       // Reset temporal accumulation
    int neighborhoodSize;   // Clamp neighborhood (1-3)
};

// ============================================================================
// MAIN
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    MotionCurr.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    
    int2 pos = int2(id.xy);
    float2 texelSize = 1.0 / float2(w, h);
    float2 uv = (float2(pos) + 0.5) * texelSize;
    
    // Current frame data
    float2 currMV = MotionCurr.Load(int3(pos, 0));
    float currConf = ConfCurr.Load(int3(pos, 0));
    
    // Build neighborhood bounds
    int k = clamp(neighborhoodSize, 1, 3);
    float2 minMV = currMV;
    float2 maxMV = currMV;
    
    // [unroll] removed for dynamic loop bounds
    for (int dy = -k; dy <= k; ++dy) {
        // [unroll] removed for dynamic loop bounds
        for (int dx = -k; dx <= k; ++dx) {
            int2 sp = clamp(pos + int2(dx, dy), int2(0,0), int2(w-1, h-1));
            float2 mv = MotionCurr.Load(int3(sp, 0));
            minMV = min(minMV, mv);
            maxMV = max(maxMV, mv);
        }
    }
    
    // Reset or invalid history
    if (resetHistory) {
        MotionOut[id.xy] = currMV;
        ConfOut[id.xy] = currConf;
        return;
    }
    
    // Reproject history
    float2 histUV = uv - currMV * texelSize;
    
    if (any(histUV < 0.0) || any(histUV > 1.0)) {
        MotionOut[id.xy] = currMV;
        ConfOut[id.xy] = currConf;
        return;
    }
    
    // Sample history
    float2 histMV = MotionHistory.SampleLevel(LinearClamp, histUV, 0);
    float histConf = ConfHistory.SampleLevel(LinearClamp, histUV, 0);
    float currLuma = LumaCurr.Load(int3(pos, 0));
    float prevLumaReproj = LumaPrev.SampleLevel(LinearClamp, histUV, 0);
    float lumaDiff = abs(currLuma - prevLumaReproj);
    
    // Clamp history to current neighborhood
    float2 clampedHist = clamp(histMV, minMV, maxMV);
    float spread = length(maxMV - minMV);
    
    // Compute blend factor AVOID BLUR:
    // "Temporal Stabilization" should just be de-jittering, not low-pass filtering.
    float alpha = historyWeight;
    float ci = saturate(confInfluence);
    float dist = length(currMV - clampedHist);
    
    // 1. If vectors are very close, SNAP to history (perfect stabilization)
    if (dist < 0.5) {
        alpha = 0.95; 
    } 
    // 2. If vectors differ significantly, Trust CURRENT (avoid blur/lag)
    else if (dist > 4.0) {
        alpha = 0.0; // Break history immediately on fast changes
    }
    // 3. In between: Blend, but favor current to keep edges sharp
    else {
        alpha = 0.2; // Weak history
    }
    
    // Confidence Check
    float confFavorHistory = saturate(0.5 + 0.5 * (histConf - currConf));
    float confScale = lerp(1.0, confFavorHistory * 1.6, ci);
    alpha *= confScale;

    float trustCurrentThreshold = 0.2 - 0.15 * ci;
    if (currConf > histConf + trustCurrentThreshold) alpha = 0.0; // Trust new clear match

    // Motion-boundary gating: reduce history influence where local vectors diverge.
    float boundaryReject = saturate((spread - 0.9) * 0.40);
    float boundaryScale = 1.0 - boundaryReject * lerp(0.55, 0.85, ci);
    alpha *= saturate(boundaryScale);

    // Reduce history pull when reprojection disagrees in luminance.
    float lumaReject = saturate((lumaDiff - 0.02) * 18.0);
    float lumaScale = 1.0 - lumaReject * lerp(0.65, 0.90, ci);
    alpha *= saturate(lumaScale);
    
    alpha = clamp(alpha, 0.0, 0.95);
    
    // Blend
    MotionOut[id.xy] = lerp(currMV, clampedHist, alpha);
    float confAlpha = alpha * (0.6 + 0.2 * ci);
    ConfOut[id.xy] = lerp(currConf, histConf, confAlpha);
}

