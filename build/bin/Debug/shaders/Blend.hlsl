Texture2D<float4> PrevColor : register(t0);
Texture2D<float4> CurrColor : register(t1);
RWTexture2D<float4> OutColor : register(u0);

SamplerState LinearClamp : register(s0);

cbuffer BlendCB : register(b0) {
  float alpha;
  float3 pad;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
  uint outW, outH;
  OutColor.GetDimensions(outW, outH);
  if (id.x >= outW || id.y >= outH) {
    return;
  }

  uint inW, inH;
  PrevColor.GetDimensions(inW, inH);

  float2 outPos = float2(id.xy) + 0.5;
  float2 inputPos = outPos * (float2(inW, inH) / float2(outW, outH));
  float2 uv = inputPos / float2(inW, inH);

  float4 prevColor = PrevColor.SampleLevel(LinearClamp, uv, 0);
  float4 currColor = CurrColor.SampleLevel(LinearClamp, uv, 0);
  OutColor[id.xy] = lerp(prevColor, currColor, alpha);
}
