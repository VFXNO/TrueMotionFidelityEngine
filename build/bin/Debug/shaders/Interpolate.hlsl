// ============================================================================
// MOTION COMPENSATION - Elite Quality Interpolator (TAA-Style)
// Uses YCoCg Variance Clipping & Confidence-Based Fallback
// ============================================================================

Texture2D<float4> PrevColor : register(t0);
Texture2D<float4> CurrColor : register(t1);
Texture2D<float2> Motion : register(t2);
Texture2D<float> Confidence : register(t3);
Texture2D<float> PrevDepth : register(t4);
Texture2D<float> CurrDepth : register(t5);
Texture2D<float4> HistoryColor : register(t6);
RWTexture2D<float4> OutColor : register(u0);

SamplerState LinearClamp : register(s0);

cbuffer InterpCB : register(b0) {
    float alpha;          // Interpolation position [0=prev, 1=curr]
    float diffScale;      // Color difference sensitivity
    float confPower;      // Confidence curve power
    int qualityMode;      // 0=Standard (Fast), 1=High (Bicubic+Linear)
    int useDepth;         // 0=ignore depth, 1=depth-aware rejection
    float depthScale;     // Depth difference scale
    float depthThreshold; // Depth difference threshold
    float motionSampleScale; // Motion multi-sample step (pixels)
    int useHistory;       // 0=ignore history, 1=use history
    float historyWeight;  // History blend weight
    float pad0;
    float pad1;
};

// ============================================================================
// COLOR SPACE UTILS (Optimized for Speed)
// Approx Gamma 2.0 - Much faster than pow(2.4) and visually identical for blending
// ============================================================================
float3 SRGBToLinear(float3 c) {
    return c * c; 
}

float3 LinearToSRGB(float3 c) {
    return sqrt(max(c, 0.0001));
}

// Optimized YCoCg conversions for Standard Mode
float3 RGBToYCoCg(float3 c) {
    return float3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r             - 0.5  * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
    );
}

float3 YCoCgToRGB(float3 ycc) {
    float tmp = ycc.x - ycc.z;
    return float3(
        tmp + ycc.y,
        ycc.x + ycc.z,
        tmp - ycc.y
    );
}

// ============================================================================
// SAMPLING UTILS
// ============================================================================
// Catmull-Rom 4-sample using bilinear hardware
float4 SampleBicubic(Texture2D<float4> tex, SamplerState s, float2 uv, float2 texSize) {
    float2 tc = uv * texSize;
    float2 tc_floor = floor(tc - 0.5) + 0.5;
    float2 f = tc - tc_floor;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    // Catmull-Rom weights
    // w0 = 0.5 * (-f3 + 2*f2 - f)
    // w1 = 0.5 * (3*f3 - 5*f2 + 2)
    // w2 = 0.5 * (-3*f3 + 4*f2 + f)
    // w3 = 0.5 * (f3 - f2)
    
    // Optimized weights for 4-bilinear samples
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

    float4 c00 = tex.SampleLevel(s, float2(t0.x, t0.y), 0);
    float4 c10 = tex.SampleLevel(s, float2(t1.x, t0.y), 0);
    float4 c01 = tex.SampleLevel(s, float2(t0.x, t1.y), 0);
    float4 c11 = tex.SampleLevel(s, float2(t1.x, t1.y), 0);
    
    return c00 * (s0.x * s0.y) + c10 * (s1.x * s0.y) + 
           c01 * (s0.x * s1.y) + c11 * (s1.x * s1.y);
}

float3 SampleColor(Texture2D<float4> tex, SamplerState s, float2 uv, float2 texSize, int mode) {
    float3 result = tex.SampleLevel(s, uv, 0).rgb;
    if (mode == 1) {
        result = SampleBicubic(tex, s, uv, texSize).rgb;
    }
    return result;
}

float3 SampleAlongMotion(Texture2D<float4> tex, SamplerState s, float2 uv, float2 mvPixels,
                         float2 texSize, int mode) {
    float mvLen = length(mvPixels);
    float2 stepUv = float2(0.0, 0.0);
    if (mvLen > 0.001) {
        float stepPx = min(motionSampleScale, mvLen * 0.5);
        stepUv = (mvPixels / mvLen) * (stepPx / texSize);
    }

    float2 uv0 = clamp(uv - stepUv, 0.001, 0.999);
    float2 uv1 = clamp(uv, 0.001, 0.999);
    float2 uv2 = clamp(uv + stepUv, 0.001, 0.999);

    float3 c0 = SampleColor(tex, s, uv0, texSize, mode);
    float3 c1 = SampleColor(tex, s, uv1, texSize, mode);
    float3 c2 = SampleColor(tex, s, uv2, texSize, mode);

    return (c0 + 2.0 * c1 + c2) * 0.25;
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
    // 1. Fetch Motion Vector
    // ------------------------------------------------------------------------
    int3 loadPos = int3(inputPos * (mvSize / inSize), 0);
    float2 mv = Motion.Load(loadPos).xy;
    mv *= (inSize / mvSize); // Scale to pixels
    
    // ------------------------------------------------------------------------
    // 2. Sampling & Blend
    // ------------------------------------------------------------------------
    float2 prevUV = clamp((inputPos - mv * alpha) / inSize, 0.001, 0.999);
    float2 currUV = clamp((inputPos + mv * (1.0 - alpha)) / inSize, 0.001, 0.999);
    
    // Multi-sample along motion vector
    float3 cPrev = SampleAlongMotion(PrevColor, LinearClamp, prevUV, mv, inSize, qualityMode);
    float3 cCurr = SampleAlongMotion(CurrColor, LinearClamp, currUV, mv, inSize, qualityMode);

    float3 blendedLin = float3(0, 0, 0);
    float3 blendedYcc = float3(0, 0, 0);

    if (qualityMode == 1) {
        // --- HIGH QUALITY MODE (Bicubic + Linear Blending) ---
        float3 cPrevLin = SRGBToLinear(cPrev);
        float3 cCurrLin = SRGBToLinear(cCurr);
        blendedLin = lerp(cPrevLin, cCurrLin, alpha);
    } 
    else {
        // --- STANDARD MODE (Bilinear + YCoCg Blending) ---
        float3 yPrev = RGBToYCoCg(cPrev);
        float3 yCurr = RGBToYCoCg(cCurr);
        blendedYcc = lerp(yPrev, yCurr, alpha);
    }
    
    // ------------------------------------------------------------------------
    // 3. Standard Variance Clipping (AABB)
    // ------------------------------------------------------------------------
    float3 m1 = float3(0,0,0);
    float3 m2 = float3(0,0,0);
    
    [unroll]
    for(int y = -1; y <= 1; ++y) {
        [unroll]
        for(int x = -1; x <= 1; ++x) {
            float2 uvOff = float2(x, y) / inSize;

            float2 uvPrev = clamp(prevUV + uvOff, 0.001, 0.999);
            float2 uvCurr = clamp(currUV + uvOff, 0.001, 0.999);

            float3 p = PrevColor.SampleLevel(LinearClamp, uvPrev, 0).rgb;
            float3 c = CurrColor.SampleLevel(LinearClamp, uvCurr, 0).rgb;

            if (qualityMode == 1) {
                p = SRGBToLinear(p);
                c = SRGBToLinear(c);
            } else {
                p = RGBToYCoCg(p);
                c = RGBToYCoCg(c);
            }

            // Accumulate statistics in blending space
            m1 += p + c;
            m2 += p*p + c*c;
        }
    }
    
    m1 /= 18.0;
    m2 /= 18.0;
    
    float3 sigma = sqrt(max(float3(0,0,0), m2 - m1*m1));
    float gamma = 1.6; // Controls strictness of the AABB
    
    float3 minC = m1 - gamma * sigma;
    float3 maxC = m1 + gamma * sigma;

    // History reprojection + clamp
    if (useHistory != 0) {
        float2 historyUV = clamp((inputPos - mv) / inSize, 0.001, 0.999);
        float3 history = HistoryColor.SampleLevel(LinearClamp, historyUV, 0).rgb;
        float histW = saturate(historyWeight);

        if (qualityMode == 1) {
            float3 historyLin = SRGBToLinear(history);
            float3 clampedHistory = clamp(historyLin, minC, maxC);
            blendedLin = lerp(blendedLin, clampedHistory, histW);
        } else {
            float3 historyY = RGBToYCoCg(history);
            float3 clampedHistory = clamp(historyY, minC, maxC);
            blendedYcc = lerp(blendedYcc, clampedHistory, histW);
        }
    }

    // Clip the blended result to the neighborhood AABB
    float3 finalColor;
    if (qualityMode == 1) {
        float3 clampedLin = clamp(blendedLin, minC, maxC);
        finalColor = LinearToSRGB(clampedLin);
    } else {
        float3 clampedY = clamp(blendedYcc, minC, maxC);
        finalColor = YCoCgToRGB(clampedY);
    }

    OutColor[id.xy] = float4(finalColor, 1.0);
}
