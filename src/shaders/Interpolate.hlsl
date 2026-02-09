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
    float2 texelSize = 1.0 / inSize;
    
    float2 mv = Motion.SampleLevel(LinearClamp, inputUv, 0).xy;
    // Motion vectors are in luma-space pixels - scale to color-space pixels
    mv *= motionSampleScale;
    float conf = pow(saturate(Confidence.SampleLevel(LinearClamp, inputUv, 0)), confPower);
    float motionMag = length(mv);
    
    // Bidirectional Warp
    // 1. Warp Prev forward to time alpha (sample Prev at x + mv * alpha)
    float2 warpedPosPrev = inputPos + mv * alpha;
    float2 warpedUvPrev = clamp(warpedPosPrev / inSize, 0.0, 0.999);
    
    // 2. Warp Curr backward to time alpha (sample Curr at x - mv * (1-alpha))
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
    
    float3 basePrev = PrevColor.SampleLevel(LinearClamp, inputUv, 0).rgb;
    float3 baseCurr = CurrColor.SampleLevel(LinearClamp, inputUv, 0).rgb;
    
    float warpError = length(warpedPrev - warpedCurr); // Compare forward/backward consistency
    float baseDiff = length(baseCurr - basePrev);
    
    float lumaCurr = Luma(baseCurr);
    float lumaPrev = Luma(basePrev);
    float lumaWarped = Luma(warpedPrev);
    float lumaDiffCurr = abs(lumaCurr - lumaPrev);
    float lumaDiffWarped = abs(lumaWarped - lumaCurr);
    
    float warpReliable = saturate(1.0 - warpError * diffScale * 6.0); // Stricter error tolerance
    warpReliable = lerp(warpReliable, conf, 0.7); // Rely more on Confidence/Consistency
    
    // Calculate how much better warping is compared to just blending
    float warpGain = saturate((baseDiff - warpError) / max(baseDiff, 0.001));
    warpGain = pow(warpGain, 1.5); // Require significant improvement to trust warp
    
    float occlusionTest = lumaDiffWarped / max(lumaDiffCurr, 0.001);
    float occlusionFactor = saturate(1.0 - (occlusionTest - 1.0) * 0.5);

    float lC = Luma(baseCurr);
    float lL = Luma(CurrColor.SampleLevel(LinearClamp, clamp(inputUv + float2(-texelSize.x, 0.0), 0.0, 0.999), 0).rgb);
    float lR = Luma(CurrColor.SampleLevel(LinearClamp, clamp(inputUv + float2(texelSize.x, 0.0), 0.0, 0.999), 0).rgb);
    float lU = Luma(CurrColor.SampleLevel(LinearClamp, clamp(inputUv + float2(0.0, -texelSize.y), 0.0, 0.999), 0).rgb);
    float lD = Luma(CurrColor.SampleLevel(LinearClamp, clamp(inputUv + float2(0.0, texelSize.y), 0.0, 0.999), 0).rgb);
    float edgeMag = abs(lR - lL) + abs(lU - lD);
    
    // User Request: ALWAYS interpellate.
    // We ignore warping errors or confidence. We trust the flow.
    float interpBlend = 1.0; 
    
    // Still respect text protection slightly if defined, but otherwise FULL MOTION
    if (textProtect > 0.0 && edgeMag > edgeThreshold && conf < 0.25) {
         interpBlend = 0.0; // Hard cutoff for text only
    }
    
    if (textProtect > 0.0 && edgeMag > edgeThreshold && conf < 0.75) {
        interpBlend *= (1.0 - textProtect * 0.75);
    }
    
    // interpBlend = saturate(interpBlend); // Already handled logic above
    
    // Smooth Bidirectional Blend
    float3 motionInterp = lerp(warpedPrev, warpedCurr, alpha);
    
    // Fallback: simply linear blend 
    float3 linearInterp = lerp(basePrev, baseCurr, alpha);
    
    // HALO SUPPRESSION:
    // Slightly relaxed to 50% range expansion to prevent harsh clamping 
    float3 minColor = min(basePrev, baseCurr);
    float3 maxColor = max(basePrev, baseCurr);
    float3 range = maxColor - minColor;
    minColor -= range * 0.5; // Was 0.2
    maxColor += range * 0.5; // Was 0.2
    motionInterp = clamp(motionInterp, minColor, maxColor);
    
    // Always use motion result unless text protection kicked in
    float3 result = lerp(linearInterp, motionInterp, interpBlend);
    
    OutColor[id.xy] = float4(saturate(result), 1.0);
}
