Texture2D<float4> Src : register(t0);
RWTexture2D<float4> OutColor : register(u0);

SamplerState LinearClamp : register(s0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
  uint outW, outH;
  OutColor.GetDimensions(outW, outH);
  if (id.x >= outW || id.y >= outH) {
    return;
  }

  float2 uv = (float2(id.xy) + 0.5) / float2(outW, outH);
  float4 color = Src.SampleLevel(LinearClamp, uv, 0);
  OutColor[id.xy] = color;
}
