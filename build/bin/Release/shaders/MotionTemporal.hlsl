// ============================================================================
// MOTION TAA v2 (Restored) - Standard Neighborhood Clamping
// "The Old Implementation" that works wonders.
// Now with user-adjustable neighborhood size.
// ============================================================================

Texture2D<float2> MotionCurr : register(t0);    // Current Raw/Smoothed Motion
Texture2D<float> ConfCurr : register(t1);
Texture2D<float2> MotionHistory : register(t2); // Previous Frame's Result
Texture2D<float> ConfHistory : register(t3);
Texture2D<float> LumaPrev : register(t4);
Texture2D<float> LumaCurr : register(t5);

RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfOut : register(u1);

SamplerState LinearClamp : register(s0);

cbuffer TemporalCB : register(b0) {
    float historyWeight; 
    float confInfluence;
    int resetHistory;
    int neighborhoodSize; // WAS pad. Range 1..4 typically.
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    MotionCurr.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height) return;
    
    float2 texelSize = 1.0 / float2(width, height);
    float2 uv = (float2(id.xy) + 0.5) * texelSize;
    
    // ------------------------------------------------------------------------
    // 1. Min/Max Neighborhood (Box Clamping)
    // ------------------------------------------------------------------------
    // Adjustable neighborhood size for variable stability.
    
    float2 currMotion = MotionCurr.Load(int3(id.xy, 0));
    float currConf = ConfCurr.Load(int3(id.xy, 0));
    
    float2 minMotion = currMotion;
    float2 maxMotion = currMotion;
    
    int k = neighborhoodSize;
    // Sanity check
    if (k < 1) k = 1;
    if (k > 5) k = 5;

    [loop]
    for(int y = -k; y <= k; ++y) {
        [loop]
        for(int x = -k; x <= k; ++x) {
            int2 nPos = clamp(int2(id.xy) + int2(x, y), int2(0,0), int2(width-1, height-1));
            float2 v = MotionCurr.Load(int3(nPos, 0));
            minMotion = min(minMotion, v);
            maxMotion = max(maxMotion, v);
        }
    }
    
    // ------------------------------------------------------------------------
    // 2. Reprojection
    // ------------------------------------------------------------------------
    float2 historyUV = uv - (currMotion * texelSize);
    
    bool validHistory = !resetHistory;
    if (any(historyUV < 0.0f) || any(historyUV > 1.0f)) {
        validHistory = false;
    }
    
    if (!validHistory) {
        MotionOut[id.xy] = currMotion;
        ConfOut[id.xy] = currConf;
        return;
    }
    
    // ------------------------------------------------------------------------
    // 3. Sample & Clamp History
    // ------------------------------------------------------------------------
    float2 historyMotion = MotionHistory.SampleLevel(LinearClamp, historyUV, 0);
    float historyConf = ConfHistory.SampleLevel(LinearClamp, historyUV, 0);
    
    // The Magic Fix: Clamp history to the current neighborhood range.
    // STRICT CLAMP RESTORED: This is essential to prevent ghosting.
    float2 clampedHistory = clamp(historyMotion, minMotion, maxMotion);
    
    // ------------------------------------------------------------------------
    // 4. Blend
    // ------------------------------------------------------------------------
    float alpha = historyWeight;
    
    // INTELLIGENT FALLOFF:
    // Only drop history if the change is large AND we are confident in the new data.
    // This prevents flickering when the camera moves fast (high dist) but the
    // estimation is noisy (low conf).
    float dist = length(currMotion - clampedHistory);
    
    if (dist > 2.0) { // Threshold for "Significant Change"
        // If confidence is high (1.0), we allow alpha to drop (accept change).
        // If confidence is low (0.0), we keep alpha high (reject noise).
        // The factor 0.3 means we drop alpha by up to 70% if confident.
        float changeCredibility = smoothstep(0.2, 0.8, currConf);
        alpha *= lerp(1.0, 0.3, changeCredibility);
    }
    
    // Confidence Influence:
    // If we are highly confident in the NEW data, we need less history.
    // If the NEW data is garbage (low conf), stick to history.
    float trustCurrent = currConf * confInfluence;
    alpha = clamp(alpha * (1.0 - trustCurrent * 0.5), 0.0, 0.99); // *0.5 to keep it stable
    
    MotionOut[id.xy] = lerp(currMotion, clampedHistory, alpha);
    
    // Confidence accumulates slowly
    ConfOut[id.xy] = lerp(currConf, historyConf, alpha);
}
