// ============================================================================
// MOTION COMPENSATION - Elite Quality Interpolator (TAA-Style)
// Uses YCoCg Variance Clipping & Confidence-Based Fallback
// ============================================================================

Texture2D<float4> PrevColor : register(t0);
Texture2D<float4> CurrColor : register(t1);
Texture2D<float2> Motion : register(t2);
Texture2D<float> Confidence : register(t3);
RWTexture2D<float4> OutColor : register(u0);

SamplerState LinearClamp : register(s0);

cbuffer InterpCB : register(b0) {
    float alpha;          // Interpolation position [0=prev, 1=curr]
    float diffScale;      // Color difference sensitivity
    float confPower;      // Confidence curve power
    float pad;
};

// ============================================================================
// COLOR SPACE UTILS (YCoCg for better Luma/Chroma separation)
// ============================================================================
float3 RGBToYCoCg(float3 c) {
    return float3(
        c.r/4.0 + c.g/2.0 + c.b/4.0,
        c.r/2.0 - c.b/2.0,
        -c.r/4.0 + c.g/2.0 - c.b/4.0
    );
}

float3 YCoCgToRGB(float3 c) {
    float Y = c.x;
    float Co = c.y;
    float Cg = c.z;
    return saturate(float3(
        Y + Co - Cg,
        Y + Cg,
        Y - Co - Cg
    ));
}

// ============================================================================
// MAIN SHADER
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutColor.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    
    // Setup coordinates
    uint inW, inH;
    PrevColor.GetDimensions(inW, inH);
    uint mvW, mvH;
    Motion.GetDimensions(mvW, mvH);
    
    float2 outSize = float2(outW, outH);
    float2 inSize = float2(inW, inH);
    float2 mvSize = float2(mvW, mvH);
    
    float2 outPos = float2(id.xy) + 0.5;
    float2 inputPos = outPos * (inSize / outSize);
    float2 inputUv = inputPos / inSize;
    
    // ------------------------------------------------------------------------
    // 1. Fetch Motion Vector & Confidence
    // ------------------------------------------------------------------------
    float2 mv = Motion.SampleLevel(LinearClamp, inputUv, 0);
    float conf = Confidence.SampleLevel(LinearClamp, inputUv, 0);
    
    mv *= (inSize / mvSize); // Scale to pixels
    
    // ------------------------------------------------------------------------
    // 2. Bidirectional Warp
    // ------------------------------------------------------------------------
    float2 prevUV = clamp((inputPos - mv * alpha) / inSize, 0.001, 0.999);
    float2 currUV = clamp((inputPos + mv * (1.0 - alpha)) / inSize, 0.001, 0.999);
    
    float3 cPrev = PrevColor.SampleLevel(LinearClamp, prevUV, 0).rgb;
    float3 cCurr = CurrColor.SampleLevel(LinearClamp, currUV, 0).rgb;
    
    // Motion Compensated Blend
    float3 warpedColor = lerp(cPrev, cCurr, alpha);
    
    // Static blend (Fallback)
    float3 staticColor = lerp(PrevColor.SampleLevel(LinearClamp, inputUv, 0).rgb,
                              CurrColor.SampleLevel(LinearClamp, inputUv, 0).rgb, 
                              alpha);
                              
    // ------------------------------------------------------------------------
    // 3. Variance Clipping (Relaxed)
    // ------------------------------------------------------------------------
    // "Soft Box" using statistical variance.
    
    float3 m1 = float3(0,0,0);
    float3 m2 = float3(0,0,0);
    
    // Sample 3x3 Neighborhood in YCoCg
    [unroll]
    for(int y = -1; y <= 1; ++y) {
        [unroll]
        for(int x = -1; x <= 1; ++x) {
            float2 uvOff = inputUv + float2(x,y) / inSize;
            float3 p = PrevColor.SampleLevel(LinearClamp, uvOff, 0).rgb;
            float3 c = CurrColor.SampleLevel(LinearClamp, uvOff, 0).rgb;
            
            float3 pY = RGBToYCoCg(p);
            float3 cY = RGBToYCoCg(c);
            
            m1 += pY + cY;
            m2 += pY*pY + cY*cY;
        }
    }
    
    m1 /= 18.0;
    m2 /= 18.0;
    
    float3 sigma = sqrt(max(float3(0,0,0), m2 - m1*m1));
    float gamma = 2.0; // WIDENED CLAMP for safety
    
    float3 minC = m1 - gamma * sigma;
    float3 maxC = m1 + gamma * sigma;
    
    // Clamp
    float3 warpedYCoCg = RGBToYCoCg(warpedColor);
    float3 clampedYCoCg = clamp(warpedYCoCg, minC, maxC);
    
    // Mix clamped with original (85% clamped) to keep some detail
    float3 finalYCoCg = lerp(warpedYCoCg, clampedYCoCg, 0.85);
    float3 finalColor = YCoCgToRGB(finalYCoCg);
    
    // ------------------------------------------------------------------------
    // 4. Safe Mode (Less Aggressive)
    // ------------------------------------------------------------------------
    // Only fall back if confidence is REALLY bad
    float safety = smoothstep(0.1, 0.4, conf); 
    
    // Ignore variance fallback for now, as it might kill texture motion
    
    finalColor = lerp(staticColor, finalColor, safety);
    
    OutColor[id.xy] = float4(finalColor, 1.0);
}
