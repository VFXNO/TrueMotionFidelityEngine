// ============================================================================
// DOWNSAMPLE LUMA R - Feature Map 2x2 box downsample (Average Pooling)
// Used for building the feature pyramid from half -> quarter -> eighth
// ============================================================================

Texture2D<float4> Src : register(t0);
Texture2D<float4> SrcFeature2 : register(t1);
Texture2D<float4> SrcFeature3 : register(t2);
RWTexture2D<float4> LumaOut : register(u0);
RWTexture2D<float4> Feature2Out : register(u1);
RWTexture2D<float4> Feature3Out : register(u2);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint outW, outH;
    LumaOut.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;

    uint inW, inH;
    Src.GetDimensions(inW, inH);
    uint2 base = id.xy * 2;

    uint2 p00 = min(base,               uint2(inW - 1, inH - 1));
    uint2 p10 = min(base + uint2(1, 0), uint2(inW - 1, inH - 1));
    uint2 p01 = min(base + uint2(0, 1), uint2(inW - 1, inH - 1));
    uint2 p11 = min(base + uint2(1, 1), uint2(inW - 1, inH - 1));

    float4 avg = Src.Load(int3(p00, 0)) + Src.Load(int3(p10, 0)) +
                 Src.Load(int3(p01, 0)) + Src.Load(int3(p11, 0));

    float4 avgF2 = SrcFeature2.Load(int3(p00, 0)) + SrcFeature2.Load(int3(p10, 0)) +
                   SrcFeature2.Load(int3(p01, 0)) + SrcFeature2.Load(int3(p11, 0));

    float4 avgF3 = SrcFeature3.Load(int3(p00, 0)) + SrcFeature3.Load(int3(p10, 0)) +
                   SrcFeature3.Load(int3(p01, 0)) + SrcFeature3.Load(int3(p11, 0));

    LumaOut[id.xy] = avg * 0.25;
    Feature2Out[id.xy] = avgF2 * 0.25;
    Feature3Out[id.xy] = avgF3 * 0.25;
}
