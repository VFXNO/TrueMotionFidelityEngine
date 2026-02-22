// ============================================================================
// FRAME INTERPOLATION - High Quality Motion-Compensated Frame Generation
// Smooth blending with occlusion handling for video content
// ============================================================================

Texture2D<float4> PrevColor : register(t0);
Texture2D<float4> CurrColor : register(t1);
Texture2D<float2> Motion : register(t2);
Texture2D<float> Confidence : register(t3);
Texture2D<float2> MotionBackward : register(t4);
Texture2D<float> ConfidenceBackward : register(t5);
Texture2D<float4> HistoryColor : register(t6);
RWTexture2D<float4> OutColor : register(u0);

SamplerState LinearClamp : register(s0);

cbuffer InterpCB : register(b0) {
    float alpha;
    float diffScale;
    float confPower;
    int qualityMode;
    int useHistory;
    float historyWeight;
    float textProtect;
    float edgeThreshold;
    float motionSampleScale;
};

static const float3 kLumaWeights = float3(0.2126, 0.7152, 0.0722);

float Luma(float3 c) {
    return dot(c, kLumaWeights);
}

float3 SampleBicubic(Texture2D<float4> tex, float2 uv, float2 texSize) {
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

    return tex.SampleLevel(LinearClamp, float2(t0.x, t0.y), 0).rgb * (s0.x * s0.y) +
           tex.SampleLevel(LinearClamp, float2(t1.x, t0.y), 0).rgb * (s1.x * s0.y) +
           tex.SampleLevel(LinearClamp, float2(t0.x, t1.y), 0).rgb * (s0.x * s1.y) +
           tex.SampleLevel(LinearClamp, float2(t1.x, t1.y), 0).rgb * (s1.x * s1.y);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutColor.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    
    uint inW, inH;
    PrevColor.GetDimensions(inW, inH);
    
    float2 outSize = float2(outW, outH);
    float2 inSize = float2(inW, inH);
    
    float2 outPos = float2(id.xy) + 0.5;
    float2 inputPos = outPos * (inSize / outSize);
    float2 inputUv = inputPos / inSize;

    // Motion vectors are in luma/tiny-space pixels - scale to color-space pixels.
    float2 mv = Motion.SampleLevel(LinearClamp, inputUv, 0).xy * motionSampleScale;
    float confSample = max(Confidence.SampleLevel(LinearClamp, inputUv, 0), 0.0);
    float conf = saturate(pow(confSample, confPower));
    float coarseMotion = saturate((motionSampleScale - 2.0) / 4.0); // ~0 at half-res MV, ~1 at tiny MV

    // Minimal pipeline stabilizer: cheap 5-tap motion smoothing to reduce tiny-field jitter.
    if (coarseMotion > 0.01) {
        uint mvW, mvH;
        Motion.GetDimensions(mvW, mvH);
        float2 mvTexel = 1.0 / float2(max(mvW, 1u), max(mvH, 1u));

        float2 mvCenter = mv;
        float2 mvSum = mvCenter * (0.7 + conf);
        float wSum = 0.7 + conf;

        float2 uvL = clamp(inputUv + float2(-mvTexel.x, 0.0), 0.0, 0.999);
        float2 uvR = clamp(inputUv + float2( mvTexel.x, 0.0), 0.0, 0.999);
        float2 uvU = clamp(inputUv + float2(0.0, -mvTexel.y), 0.0, 0.999);
        float2 uvD = clamp(inputUv + float2(0.0,  mvTexel.y), 0.0, 0.999);

        float2 mvL = Motion.SampleLevel(LinearClamp, uvL, 0).xy * motionSampleScale;
        float2 mvR = Motion.SampleLevel(LinearClamp, uvR, 0).xy * motionSampleScale;
        float2 mvU = Motion.SampleLevel(LinearClamp, uvU, 0).xy * motionSampleScale;
        float2 mvD = Motion.SampleLevel(LinearClamp, uvD, 0).xy * motionSampleScale;

        float confL = saturate(Confidence.SampleLevel(LinearClamp, uvL, 0));
        float confR = saturate(Confidence.SampleLevel(LinearClamp, uvR, 0));
        float confU = saturate(Confidence.SampleLevel(LinearClamp, uvU, 0));
        float confD = saturate(Confidence.SampleLevel(LinearClamp, uvD, 0));

        float simL = exp(-dot(mvL - mvCenter, mvL - mvCenter) / 12.0);
        float simR = exp(-dot(mvR - mvCenter, mvR - mvCenter) / 12.0);
        float simU = exp(-dot(mvU - mvCenter, mvU - mvCenter) / 12.0);
        float simD = exp(-dot(mvD - mvCenter, mvD - mvCenter) / 12.0);

        float wL = (0.1 + 0.9 * confL) * simL;
        float wR = (0.1 + 0.9 * confR) * simR;
        float wU = (0.1 + 0.9 * confU) * simU;
        float wD = (0.1 + 0.9 * confD) * simD;

        mvSum += mvL * wL + mvR * wR + mvU * wU + mvD * wD;
        wSum += wL + wR + wU + wD;
        float2 mvSmooth = mvSum / max(wSum, 1e-4);
        mv = lerp(mvCenter, mvSmooth, 0.65);
    }

    float2 warpedPosPrev = inputPos + mv * alpha;
    float2 warpedUvPrev = clamp(warpedPosPrev / inSize, 0.0, 0.999);
    
    float2 warpedPosCurr = inputPos - mv * (1.0 - alpha);
    float2 warpedUvCurr = clamp(warpedPosCurr / inSize, 0.0, 0.999);
    
    float3 warpedPrev, warpedCurr;
    
    if (qualityMode == 1) {
        warpedPrev = SampleBicubic(PrevColor, warpedUvPrev, inSize);
        warpedCurr = SampleBicubic(CurrColor, warpedUvCurr, inSize);
    } else {
        warpedPrev = PrevColor.SampleLevel(LinearClamp, warpedUvPrev, 0).rgb;
        warpedCurr = CurrColor.SampleLevel(LinearClamp, warpedUvCurr, 0).rgb;
    }

    // Minimal anti-ghosting using forward/backward consistency.
    float reliability = conf;
    if (coarseMotion > 0.01) {
        float2 backMv = MotionBackward.SampleLevel(LinearClamp, warpedUvPrev, 0).xy * motionSampleScale;
        float backConf = saturate(ConfidenceBackward.SampleLevel(LinearClamp, warpedUvPrev, 0));
        float consistencyErr = length(mv + backMv);
        float consistency = 1.0 - smoothstep(0.5, 4.0, consistencyErr);
        reliability *= lerp(0.5, 1.0, backConf) * consistency;
    }

    float motionMag = length(mv);
    float ghostRisk = (1.0 - reliability) * saturate(motionMag * 0.08) * coarseMotion;
    float3 blended = lerp(warpedPrev, warpedCurr, alpha);
    float fallbackAlpha = smoothstep(0.30, 0.70, alpha);
    float3 fallback = lerp(warpedPrev, warpedCurr, fallbackAlpha);
    float3 result = lerp(blended, fallback, ghostRisk);
    
    OutColor[id.xy] = float4(saturate(result), 1.0);
}
