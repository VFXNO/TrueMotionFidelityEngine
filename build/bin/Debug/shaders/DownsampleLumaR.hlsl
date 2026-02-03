Texture2D<float> Src : register(t0);
RWTexture2D<float> LumaOut : register(u0);

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
  uint outW, outH;
  LumaOut.GetDimensions(outW, outH);
  if (id.x >= outW || id.y >= outH) {
    return;
  }

  uint inW, inH;
  Src.GetDimensions(inW, inH);
  uint2 base = id.xy * 2;

  uint2 p0 = min(base, uint2(inW - 1, inH - 1));
  uint2 p1 = min(base + uint2(1, 0), uint2(inW - 1, inH - 1));
  uint2 p2 = min(base + uint2(0, 1), uint2(inW - 1, inH - 1));
  uint2 p3 = min(base + uint2(1, 1), uint2(inW - 1, inH - 1));

  float luma = 0.0;
  luma += Src.Load(int3(p0, 0));
  luma += Src.Load(int3(p1, 0));
  luma += Src.Load(int3(p2, 0));
  luma += Src.Load(int3(p3, 0));

  LumaOut[id.xy] = luma * 0.25;
}
