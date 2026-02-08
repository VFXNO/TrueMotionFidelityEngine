// ============================================================================
// GAME FRAME INTERPOLATION - Professional Grade
// Forward-Backward Consistency + Occlusion-Aware Splatting
// ============================================================================

Texture2D<float4> PrevColor : register(t0);
Texture2D<float4> CurrColor : register(t1);
Texture2D<float2> Motion : register(t2);
Texture2D<float> Confidence : register(t3);
Texture2D<float> PrevDepth : register(t4);
Texture2D<float> CurrDepth : register(t5);
Texture2D<float4> HistoryColor : register(t6);
RWTexture2D<float4> OutColor : register(u0);

SamplerState LinearClamp : register(s0);

cbuffer InterpCB : register(b0) {
    float alpha;          // Interpolation position [0=prev, 1=curr]
    float diffScale;      // Color difference sensitivity
    float confPower;      // Confidence curve power
    int qualityMode;      // 0=Standard, 1=High (Bicubic)
    int useDepth;         
    float depthScale;     
    float depthThreshold; 
    float motionSampleScale;
    int useHistory;       
    float historyWeight;  
    float textProtect;
    float edgeThreshold;
};

float4 SampleBicubic(Texture2D<float4> tex, float2 uv, float2 texSize);
static const float3 kLumaWeights = float3(0.2126, 0.7152, 0.0722);

float CandidateWarpScore(
    float2 candMV,
    float2 inputPos,
    float2 inSize,
    float alpha,
    float3 basePrev,
    float3 baseCurr) {
    float2 prevUV = clamp((inputPos + candMV * alpha) / inSize, 0.001, 0.999);
    float2 currUV = clamp((inputPos - candMV * (1.0 - alpha)) / inSize, 0.001, 0.999);

    float3 p = PrevColor.SampleLevel(LinearClamp, prevUV, 0).rgb;
    float3 c = CurrColor.SampleLevel(LinearClamp, currUV, 0).rgb;

    float lp = dot(p, kLumaWeights);
    float lc = dot(c, kLumaWeights);
    float lPrev = dot(basePrev, kLumaWeights);
    float lCurr = dot(baseCurr, kLumaWeights);

    float symmetryErr = abs(lp - lCurr) + abs(lc - lPrev);
    float pairErr = abs(lp - lc);
    return symmetryErr + pairErr * 0.35;
}

float3 SampleColor(Texture2D<float4> tex, float2 uv, float2 texSize, int mode) {
    float3 sampleColor = tex.SampleLevel(LinearClamp, uv, 0).rgb;
    if (mode != 0) {
        sampleColor = SampleBicubic(tex, uv, texSize).rgb;
    }
    return sampleColor;
}

// ============================================================================
// CATMULL-ROM BICUBIC (Sharp, minimal ringing)
// ============================================================================
float4 SampleBicubic(Texture2D<float4> tex, float2 uv, float2 texSize) {
    float2 tc = uv * texSize;
    float2 tc_floor = floor(tc - 0.5) + 0.5;
    float2 f = tc - tc_floor;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = f2 - 0.5 * (f3 + f);
    float2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    float2 w3 = 0.5 * (f3 - f2);
    float2 w2 = 1.0 - w0 - w1 - w3;

    float2 s0 = w0 + w1;
    float2 s1 = w2 + w3;
    float2 f0 = w1 / s0;
    float2 f1 = w3 / s1;

    float2 t0 = (tc_floor - 1.0 + f0) / texSize;
    float2 t1 = (tc_floor + 1.0 + f1) / texSize;

    return tex.SampleLevel(LinearClamp, float2(t0.x, t0.y), 0) * (s0.x * s0.y) +
           tex.SampleLevel(LinearClamp, float2(t1.x, t0.y), 0) * (s1.x * s0.y) +
           tex.SampleLevel(LinearClamp, float2(t0.x, t1.y), 0) * (s0.x * s1.y) +
           tex.SampleLevel(LinearClamp, float2(t1.x, t1.y), 0) * (s1.x * s1.y);
}

// ============================================================================
// MAIN SHADER
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutColor.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    
    // Get dimensions
    uint inW, inH;
    PrevColor.GetDimensions(inW, inH);
    uint mvW, mvH;
    Motion.GetDimensions(mvW, mvH);
    
    float2 outSize = float2(outW, outH);
    float2 inSize = float2(inW, inH);
    float2 mvSize = float2(mvW, mvH);
    
    // Calculate positions
    float2 outPos = float2(id.xy) + 0.5;
    float2 inputPos = outPos * (inSize / outSize);
    float2 inputUv = inputPos / inSize;
    
    // ========================================================================
    // 1. FETCH MOTION VECTOR
    // ========================================================================
    float2 mvUv = inputUv;
    float2 mv = Motion.SampleLevel(LinearClamp, mvUv, 0).xy;
    mv *= (inSize / mvSize);
    float conf = pow(saturate(Confidence.SampleLevel(LinearClamp, mvUv, 0)), confPower);
    
    // ========================================================================
    // 2. MOTION-SELECTIVE INTERPOLATION
    // Only moving regions are warped/interpolated; static text/HUD remains crisp.
    // ========================================================================
    float3 basePrev = PrevColor.SampleLevel(LinearClamp, inputUv, 0).rgb;
    float3 baseCurr = CurrColor.SampleLevel(LinearClamp, inputUv, 0).rgb;
    float3 baseStable = lerp(basePrev, baseCurr, alpha);
    float3 baseNearest = (alpha < 0.5) ? basePrev : baseCurr;

    float motionMag = length(mv);
    float motionThreshold = max(0.35, motionSampleScale * 0.55);
    float motionMask = smoothstep(motionThreshold * 0.60, motionThreshold, motionMag);

    float3 baseDiff = abs(baseCurr - basePrev);
    float diffMetric = max(baseDiff.r, max(baseDiff.g, baseDiff.b));
    float diffMask = smoothstep(0.008, 0.030 * max(0.5, diffScale), diffMetric);

    float confMask = smoothstep(0.15, 0.55, conf);
    float interpMask = motionMask * max(diffMask, confMask);
    float interpWeightRaw = saturate(interpMask);

    // Static text lock (smooth, no binary snapping):
    // strong edges + low motion + low frame difference should remain unwarped.
    float2 texelUv = 1.0 / inSize;
    float2 rightUv = clamp(inputUv + float2(texelUv.x, 0.0), 0.001, 0.999);
    float2 downUv = clamp(inputUv + float2(0.0, texelUv.y), 0.001, 0.999);
    float3 currRight = CurrColor.SampleLevel(LinearClamp, rightUv, 0).rgb;
    float3 currDown = CurrColor.SampleLevel(LinearClamp, downUv, 0).rgb;
    float lumaC = dot(baseCurr, kLumaWeights);
    float edgeMetric = abs(dot(currRight, kLumaWeights) - lumaC) +
                       abs(dot(currDown,  kLumaWeights) - lumaC);
    float edgeT = max(0.004, edgeThreshold * 0.35 + 0.004);
    float edgeMask = smoothstep(edgeT, edgeT * 2.5, edgeMetric);
    float staticMask = (1.0 - motionMask) * (1.0 - diffMask);
    float textLockStrength = 0.30 + 0.70 * saturate(textProtect);
    float textLock = saturate(textLockStrength * edgeMask * staticMask);

    float interpWeight = interpWeightRaw * (1.0 - textLock);
    interpWeight *= smoothstep(0.015, 0.06, interpWeightRaw);
    float3 staticBase = lerp(baseNearest, baseStable, textLock * 0.75);

    // Fast path for static pixels: skip expensive warping/bicubic sampling.
    if (interpWeight < 0.006 || alpha <= 0.001 || alpha >= 0.999) {
        OutColor[id.xy] = float4(saturate(staticBase), 1.0);
        return;
    }

    // ========================================================================
    // 3. BIDIRECTIONAL WARPING (only for moving pixels)
    // ========================================================================
    // AI-like multi-hypothesis motion selection:
    // evaluate local consensus candidates and choose the most photometrically consistent warp.
    float2 selectedMV = mv;
    if (qualityMode != 0 && interpWeightRaw > 0.04) {
        float2 mvTexel = 1.0 / mvSize;
        float2 mvScale = (inSize / mvSize);

        float2 mvL = Motion.SampleLevel(LinearClamp, mvUv + float2(-mvTexel.x, 0.0), 0).xy * mvScale;
        float2 mvR = Motion.SampleLevel(LinearClamp, mvUv + float2( mvTexel.x, 0.0), 0).xy * mvScale;
        float2 mvU = Motion.SampleLevel(LinearClamp, mvUv + float2(0.0, -mvTexel.y), 0).xy * mvScale;
        float2 mvD = Motion.SampleLevel(LinearClamp, mvUv + float2(0.0,  mvTexel.y), 0).xy * mvScale;

        float cL = pow(saturate(Confidence.SampleLevel(LinearClamp, mvUv + float2(-mvTexel.x, 0.0), 0)), confPower);
        float cR = pow(saturate(Confidence.SampleLevel(LinearClamp, mvUv + float2( mvTexel.x, 0.0), 0)), confPower);
        float cU = pow(saturate(Confidence.SampleLevel(LinearClamp, mvUv + float2(0.0, -mvTexel.y), 0)), confPower);
        float cD = pow(saturate(Confidence.SampleLevel(LinearClamp, mvUv + float2(0.0,  mvTexel.y), 0)), confPower);

        float w0 = max(0.05, conf);
        float ws = w0 + cL + cR + cU + cD;
        float2 consensusMV = (mv * w0 + mvL * cL + mvR * cR + mvU * cU + mvD * cD) / max(ws, 1e-4);

        float2 consDelta = consensusMV - mv;
        float consLimit = max(0.75, motionSampleScale * 1.6);
        consensusMV = mv + clamp(consDelta, -consLimit, consLimit);

        float2 cand0 = mv;
        float2 cand1 = lerp(mv, consensusMV, 0.50);
        float2 cand2 = consensusMV;

        float s0 = CandidateWarpScore(cand0, inputPos, inSize, alpha, basePrev, baseCurr);
        float s1 = CandidateWarpScore(cand1, inputPos, inSize, alpha, basePrev, baseCurr) + length(cand1 - mv) * 0.008;
        float s2 = CandidateWarpScore(cand2, inputPos, inSize, alpha, basePrev, baseCurr) + length(cand2 - mv) * 0.012;

        selectedMV = cand0;
        float bestS = s0;
        if (s1 < bestS) {
            bestS = s1;
            selectedMV = cand1;
        }
        if (s2 < bestS) {
            selectedMV = cand2;
        }
    }

    float2 prevWarpPos = inputPos + selectedMV * alpha;
    float2 currWarpPos = inputPos - selectedMV * (1.0 - alpha);
    
    float2 prevUV = clamp(prevWarpPos / inSize, 0.001, 0.999);
    float2 currUV = clamp(currWarpPos / inSize, 0.001, 0.999);
    
    // Respect quality mode: 0 = bilinear (faster), 1 = bicubic (sharper)
    float3 cPrev = SampleColor(PrevColor, prevUV, inSize, qualityMode);
    float3 cCurr = SampleColor(CurrColor, currUV, inSize, qualityMode);
    float3 warped = lerp(cPrev, cCurr, alpha);

    // Warp validity gate:
    // only trust motion warping if it improves Prev/Curr alignment over no-warp.
    float zeroErr =
        abs(dot(basePrev, kLumaWeights) - dot(baseCurr, kLumaWeights));
    float warpErr =
        0.5 * (abs(dot(cPrev, kLumaWeights) - dot(baseCurr, kLumaWeights)) +
               abs(dot(cCurr, kLumaWeights) - dot(basePrev, kLumaWeights)));
    float warpGain = (zeroErr - warpErr) / max(zeroErr, 0.01);
    float warpTrust = smoothstep(0.01, 0.16, warpGain);

    float finalWeight = interpWeight * warpTrust;
    float3 result = lerp(staticBase, warped, finalWeight);

    // ========================================================================
    // OUTPUT
    // ========================================================================
    OutColor[id.xy] = float4(saturate(result), 1.0);
}
