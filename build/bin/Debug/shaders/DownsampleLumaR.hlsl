// ============================================================================
// DOWNSAMPLE LUMA R - Luma to Luma 2x2 downsample
// Simple box filter for pyramid levels
// ============================================================================

Texture2D<float> Src : register(t0);
RWTexture2D<float> LumaOut : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint outW, outH;
    LumaOut.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    
    uint inW, inH;
    Src.GetDimensions(inW, inH);
    uint2 base = id.xy * 2;
    
    // 2x2 box filter
    uint2 p0 = min(base, uint2(inW-1, inH-1));
    uint2 p1 = min(base + uint2(1,0), uint2(inW-1, inH-1));
    uint2 p2 = min(base + uint2(0,1), uint2(inW-1, inH-1));
    uint2 p3 = min(base + uint2(1,1), uint2(inW-1, inH-1));
    
    float avg = Src.Load(int3(p0,0)) + Src.Load(int3(p1,0)) +
                Src.Load(int3(p2,0)) + Src.Load(int3(p3,0));
    
    LumaOut[id.xy] = avg * 0.25;
}
