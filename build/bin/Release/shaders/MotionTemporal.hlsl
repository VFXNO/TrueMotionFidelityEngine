// ============================================================================
// MOTION TEMPORAL - Strong Temporal Smoothing
// Aggressive temporal blending for smooth, jitter-free motion vectors
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
    float historyWeight;
    float confInfluence;
    int resetHistory;
    int neighborhoodSize;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    MotionCurr.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    
    int2 pos = int2(id.xy);
    float2 texelSize = 1.0 / float2(w, h);
    float2 uv = (float2(pos) + 0.5) * texelSize;
    
    float2 currMV = MotionCurr.Load(int3(pos, 0));
    float currConf = ConfCurr.Load(int3(pos, 0));
    
    int k = clamp(neighborhoodSize, 1, 3);
    float2 minMV = currMV;
    float2 maxMV = currMV;
    
    for (int dy = -k; dy <= k; ++dy) {
        for (int dx = -k; dx <= k; ++dx) {
            int2 sp = clamp(pos + int2(dx, dy), int2(0, 0), int2(w - 1, h - 1));
            float2 mv = MotionCurr.Load(int3(sp, 0));
            minMV = min(minMV, mv);
            maxMV = max(maxMV, mv);
        }
    }
    
    if (resetHistory) {
        MotionOut[id.xy] = currMV;
        ConfOut[id.xy] = currConf;
        return;
    }
    
    float2 histUV = uv - currMV * texelSize;
    
    if (any(histUV < 0.0) || any(histUV > 1.0)) {
        MotionOut[id.xy] = currMV;
        ConfOut[id.xy] = currConf;
        return;
    }
    
    float2 histMV = MotionHistory.SampleLevel(LinearClamp, histUV, 0);
    float histConf = ConfHistory.SampleLevel(LinearClamp, histUV, 0);
    
    float currLuma = LumaCurr.Load(int3(pos, 0));
    float prevLuma = LumaPrev.SampleLevel(LinearClamp, histUV, 0);
    float lumaDiff = abs(currLuma - prevLuma);
    
    float2 clampedHist = clamp(histMV, minMV, maxMV);
    float spread = length(maxMV - minMV);
    
    // Adaptive blend: use clamped history to prevent drift accumulation
    // Strong temporal smoothing when motion is consistent, weaker when changing
    float mvDist = length(currMV - clampedHist);
    float blend = historyWeight;
    
    if (mvDist < 2.0) {
        // Very consistent motion — moderate temporal filtering
        blend = lerp(blend, 0.88, 0.80);
    } else if (mvDist < 8.0) {
        // Moderately consistent — lighter smoothing
        blend = lerp(blend, 0.80, 0.65);
    } else if (mvDist > 40.0) {
        // Large change — likely scene cut or fast camera move, drop history
        blend *= 0.15;
    }
    
    // Luma change detection: reduce history when content changes significantly
    float lumaBlendFactor = saturate(1.0 - lumaDiff * 10.0);
    blend *= lerp(0.4, 1.0, lumaBlendFactor);
    
    blend = clamp(blend, 0.40, 0.88); // Faster adaptation, cap at 88%

    // Use CLAMPED history to prevent drift while maintaining smoothness
    float2 resultMV = lerp(currMV, clampedHist, blend);
    float resultConf = lerp(currConf, histConf, blend * 0.7);
    
    MotionOut[id.xy] = resultMV;
    ConfOut[id.xy] = resultConf;
}
