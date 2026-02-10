// ============================================================================
// FRAME INTERPOLATION - High Quality Motion-Compensated Frame Generation
// Smooth blending with occlusion handling for video content
// ============================================================================

Texture2D<float4> PrevColor : register(t0);
Texture2D<float4> CurrColor : register(t1);
Texture2D<float2> Motion : register(t2);
Texture2D<float> Confidence : register(t3);
Texture2D<float4> HistoryColor : register(t4);
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
    
    float2 mv = Motion.SampleLevel(LinearClamp, inputUv, 0).xy;
    // Motion vectors are in luma-space pixels - scale to color-space pixels
    mv *= motionSampleScale;
    
    // Pure warp path: no linear-frame fallback blending.
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

    float3 result = lerp(warpedPrev, warpedCurr, alpha);
    
    OutColor[id.xy] = float4(saturate(result), 1.0);
}
