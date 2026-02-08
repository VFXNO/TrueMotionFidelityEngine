// ============================================================================
// DOWNSAMPLE LUMA - RGB to Luma with 2x2 downsample
// Clean conversion preserving detail
// ============================================================================

Texture2D<float4> Src : register(t0);
RWTexture2D<float> LumaOut : register(u0);

// Rec.709 luma weights
static const float3 kLumaWeights = float3(0.2126, 0.7152, 0.0722);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint outW, outH;
    LumaOut.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    
    uint inW, inH;
    Src.GetDimensions(inW, inH);
    uint2 base = id.xy * 2;
    
    // Sample 2x2 block
    uint2 p0 = min(base, uint2(inW-1, inH-1));
    uint2 p1 = min(base + uint2(1,0), uint2(inW-1, inH-1));
    uint2 p2 = min(base + uint2(0,1), uint2(inW-1, inH-1));
    uint2 p3 = min(base + uint2(1,1), uint2(inW-1, inH-1));
    
    float l0 = dot(Src.Load(int3(p0, 0)).rgb, kLumaWeights);
    float l1 = dot(Src.Load(int3(p1, 0)).rgb, kLumaWeights);
    float l2 = dot(Src.Load(int3(p2, 0)).rgb, kLumaWeights);
    float l3 = dot(Src.Load(int3(p3, 0)).rgb, kLumaWeights);
    
    // Stable box average to avoid introducing synthetic motion on static UI/text.
    float avg = (l0 + l1 + l2 + l3) * 0.25;
    LumaOut[id.xy] = avg;
}
