// ============================================================================
// MOTION TEMPORAL v2 - Motion-Compensated Temporal Accumulation
//
// Key improvements:
//   1. AABB neighborhood clamping: clamps history to the min/max of the
//      current-frame neighborhood to prevent ghosting/trailing
//   2. Motion-compensated reprojection: reads history at the warped position
//   3. Adaptive blend weight based on reprojection quality + scene change
//   4. Smooth blend curves instead of hard thresholds
// ============================================================================

Texture2D<float2> MotionCurr    : register(t0);
Texture2D<float>  ConfCurr      : register(t1);
Texture2D<float2> MotionHistory : register(t2);
Texture2D<float>  ConfHistory   : register(t3);
Texture2D<float>  LumaPrev      : register(t4);
Texture2D<float>  LumaCurr      : register(t5);

RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float>  ConfOut   : register(u1);

SamplerState LinearClamp : register(s0);

cbuffer TemporalCB : register(b0) {
    float historyWeight;
    float confInfluence;
    int   resetHistory;
    int   neighborhoodSize;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    MotionCurr.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int2 pos = int2(id.xy);
    float2 texelSize = 1.0 / float2(w, h);
    float2 uv = (float2(pos) + 0.5) * texelSize;

    float2 currMV   = MotionCurr.Load(int3(pos, 0));
    float  currConf = ConfCurr.Load(int3(pos, 0));

    // On reset (first frame), just pass through
    if (resetHistory != 0) {
        MotionOut[id.xy] = currMV;
        ConfOut[id.xy]   = currConf;
        return;
    }

    // --- Build AABB from current-frame neighborhood ---
    int k = clamp(neighborhoodSize, 1, 3);
    float2 minMV = currMV;
    float2 maxMV = currMV;
    float  minConf = currConf;
    float  maxConf = currConf;

    [loop] for (int dy = -k; dy <= k; ++dy) {
        [loop] for (int dx = -k; dx <= k; ++dx) {
            int2 sp = clamp(pos + int2(dx, dy), int2(0, 0), int2(w - 1, h - 1));
            float2 mv = MotionCurr.Load(int3(sp, 0));
            float  cf = ConfCurr.Load(int3(sp, 0));
            minMV   = min(minMV, mv);
            maxMV   = max(maxMV, mv);
            minConf = min(minConf, cf);
            maxConf = max(maxConf, cf);
        }
    }

    // --- Reprojection: read history at motion-compensated position ---
    float2 histUV = uv - currMV * texelSize;

    // Out of bounds -> use current frame only
    if (any(histUV < 0.0) || any(histUV > 1.0)) {
        MotionOut[id.xy] = currMV;
        ConfOut[id.xy]   = currConf;
        return;
    }

    float2 histMV   = MotionHistory.SampleLevel(LinearClamp, histUV, 0);
    float  histConf = ConfHistory.SampleLevel(LinearClamp, histUV, 0);

    // --- AABB clamp: prevent ghosting by clamping history to current neighborhood ---
    float2 clampedHistMV = clamp(histMV, minMV, maxMV);
    float  clampDist = length(histMV - clampedHistMV);

    // --- Scene change detection via luma comparison ---
    float currLuma = LumaCurr.Load(int3(pos, 0));
    float prevLuma = LumaPrev.SampleLevel(LinearClamp, histUV, 0);
    float lumaDiff = abs(currLuma - prevLuma);
    float sceneChange = smoothstep(0.08, 0.25, lumaDiff);

    // --- Adaptive blend weight ---
    // Start from the user-controlled historyWeight
    float blend = historyWeight;

    // Reduce history influence when:
    //  - The clamped value is far from the unclamped (disocclusion / new content)
    //  - Scene change detected
    //  - History confidence is low
    float clampPenalty = smoothstep(0.5, 4.0, clampDist);
    blend *= (1.0 - clampPenalty * 0.8);
    blend *= (1.0 - sceneChange * 0.9);
    blend *= lerp(0.5, 1.0, histConf);

    // Boost history when motion is very consistent (jitter suppression)
    float mvConsistency = exp(-dot(currMV - clampedHistMV, currMV - clampedHistMV) / 2.0);
    blend = lerp(blend, min(blend + 0.3, 0.95), mvConsistency * confInfluence);

    blend = clamp(blend, 0.0, 0.95);

    // --- Blend using the CLAMPED history (safe against ghosting) ---
    float2 resultMV   = lerp(currMV,   clampedHistMV, blend);
    float  resultConf = lerp(currConf, histConf,       blend * 0.7);

    MotionOut[id.xy] = resultMV;
    ConfOut[id.xy]   = resultConf;
}
