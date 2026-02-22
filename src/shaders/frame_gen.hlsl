// ============================================================================
// FRAME_GEN - Legacy/reference shader (unused by main pipeline)
// Kept for compatibility with test_app
// ============================================================================

cbuffer Params : register(b0) {
    float width;
    float height;
    float alpha;
    float motionScale;
    int   debugMode;
    float debugScale;
    float confidenceThreshold;
    int   motionModel;
};

Texture2D<float4> TexPrev      : register(t0);
Texture2D<float4> TexCurr      : register(t1);
Texture2D<float2> TexFlowInput : register(t2);

RWTexture2D<float2> OutFlow  : register(u0);
RWTexture2D<float4> OutFrame : register(u0);

SamplerState LinearClamp : register(s0);

float Luma(float3 color) {
    return dot(color, float3(0.299, 0.587, 0.114));
}

[numthreads(8, 8, 1)]
void CSOpticalFlow(uint3 id : SV_DispatchThreadID) {
    if (id.x >= (uint)width || id.y >= (uint)height) return;
    int2 pos = int2(id.xy);
    float centerLuma = Luma(TexPrev.Load(int3(pos, 0)).rgb);
    int radius = 4;
    float minDiff = 1e4;
    int2 bestVec = int2(0, 0);
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            int2 sp = pos + int2(x, y);
            if (sp.x < 0 || sp.x >= (int)width || sp.y < 0 || sp.y >= (int)height) continue;
            float diff = abs(Luma(TexCurr.Load(int3(sp, 0)).rgb) - centerLuma);
            if (diff < minDiff) { minDiff = diff; bestVec = int2(x, y); }
        }
    }
    OutFlow[pos] = float2(bestVec);
}

[numthreads(8, 8, 1)]
void CSInterpolate(uint3 id : SV_DispatchThreadID) {
    if (id.x >= (uint)width || id.y >= (uint)height) return;
    int2 pos = int2(id.xy);
    float2 flow = TexFlowInput.Load(int3(pos, 0));
    float2 uv = (float2(pos) + 0.5) / float2(width, height);
    float2 texelSize = 1.0 / float2(width, height);
    float2 flowVec = flow * texelSize;

    // Pure warp: forward-warp prev, backward-warp curr, alpha-weighted
    float2 uvPrev = uv - flowVec * alpha;
    float2 uvCurr = uv + flowVec * (1.0 - alpha);
    float3 warpedPrev = TexPrev.SampleLevel(LinearClamp, clamp(uvPrev, 0.0, 0.999), 0).rgb;
    float3 warpedCurr = TexCurr.SampleLevel(LinearClamp, clamp(uvCurr, 0.0, 0.999), 0).rgb;
    float3 result = lerp(warpedPrev, warpedCurr, alpha);
    OutFrame[pos] = float4(result, 1.0);
}

[numthreads(8, 8, 1)]
void CSDebug(uint3 id : SV_DispatchThreadID) {
    if (id.x >= (uint)width || id.y >= (uint)height) return;
    int2 pos = int2(id.xy);
    if (debugMode == 1) {
        float2 flow = TexFlowInput.Load(int3(pos, 0));
        float2 vis = (flow * debugScale) * 0.5 + 0.5;
        OutFrame[pos] = float4(vis.x, vis.y, 0.5, 1.0);
    } else if (debugMode == 2) {
        float4 c0 = TexPrev.Load(int3(pos, 0));
        float4 c1 = TexCurr.Load(int3(pos, 0));
        float diff = length(c0.rgb - c1.rgb);
        OutFrame[pos] = float4(diff * debugScale, 0, 0, 1);
    } else {
        OutFrame[pos] = TexCurr.Load(int3(pos, 0));
    }
}
