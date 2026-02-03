Texture2D<float4> Src : register(t0);
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

  // PACING FIX: Reverting to 2x2 Downsample
  // Even if GPU usage is low, memory bandwidth latency can cause missed VSyncs at full res.
  // Downsampling is the safest way to guarantee consistent frame times.
  uint2 p0 = min(base, uint2(inW - 1, inH - 1));
  uint2 p1 = min(base + uint2(1, 0), uint2(inW - 1, inH - 1));
  uint2 p2 = min(base + uint2(0, 1), uint2(inW - 1, inH - 1));
  uint2 p3 = min(base + uint2(1, 1), uint2(inW - 1, inH - 1));

  float3 c0 = Src.Load(int3(p0, 0)).rgb;
  float3 c1 = Src.Load(int3(p1, 0)).rgb;
  float3 c2 = Src.Load(int3(p2, 0)).rgb;
  float3 c3 = Src.Load(int3(p3, 0)).rgb;

  float3 weights = float3(0.2126, 0.7152, 0.0722);
  float l0 = dot(c0, weights);
  float l1 = dot(c1, weights);
  float l2 = dot(c2, weights);
  float l3 = dot(c3, weights);

  // MAX-POOLING HYBRID:
  // For standard downsampling, we use Average.
  // BUT for tracking small objects (stars, particles), Average destroys them.
  // We compute both Average and Max.
  float avg = (l0 + l1 + l2 + l3) * 0.25;
  float maximum = max(max(l0, l1), max(l2, l3));
  float minimum = min(min(l0, l1), min(l2, l3));
  
  // Contrast Detection:
  // If the 2x2 block has high contrast (object vs background), prefer the EXTREME value 
  // (whichever differs most from the neighborhood average).
  // This preserves "bright stars on black sky" or "dark birds on white sky".
  
  float contrast = maximum - minimum;
  float luma;
  
  if (contrast > 0.1) {
     // High contrast block - likely an edge or small object.
     // Check which value is statistically "interesting" (the outlier).
     // Ideally we'd compare to the wider neighborhood, but we only have 2x2 here.
     // Heuristic: Preserve the feature that has the 'stronger' signal energy? 
     // Simple Max-Pooling is robust for bright-on-dark. 
     // Simple Min-Pooling is robust for dark-on-bright.
     
     // Let's use a "Soft Max" approach to bias towards details.
     // We linearly blend between Average and Max/Min based on brightness.
     // Actually, let's just use Max-Pooling for now as it's the standard fix for "disappearing small objects".
     // But wait, dark objects on bright background (crows in sky) will disappear with Max pooling.
     
     // Better Heuristic: "Extremum Pooling"
     // Whichever is further from 0.5 (mid-grey) wins? No.
     // Whichever is further from the *Average* wins.
     float distMax = abs(maximum - avg);
     float distMin = abs(minimum - avg);
     
     luma = (distMax > distMin) ? maximum : minimum;
  } else {
     luma = avg;
  }
  
  LumaOut[id.xy] = luma;
}
