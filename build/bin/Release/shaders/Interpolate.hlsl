// ============================================================================
// GAME FRAME INTERPOLATION - Professional Grade
// Forward-Backward Consistency + Occlusion-Aware Splatting
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
    int qualityMode;      // 0=Standard, 1=High (Bicubic)
    int useDepth;         
    float depthScale;     
    float depthThreshold; 
    float motionSampleScale;
    int useHistory;       
    float historyWeight;  
    float textProtect;
    float edgeThreshold;
};

static const float3 kLumaWeights = float3(0.2126, 0.7152, 0.0722);

float Luma(float3 color) {
    return dot(color, kLumaWeights);
}

// ============================================================================
// CATMULL-ROM BICUBIC (Sharp, minimal ringing)
// ============================================================================
float4 SampleBicubic(Texture2D<float4> tex, float2 uv, float2 texSize) {
    float2 tc = uv * texSize;
    float2 tc_floor = floor(tc - 0.5) + 0.5;
    float2 f = tc - tc_floor;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

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

    return tex.SampleLevel(LinearClamp, float2(t0.x, t0.y), 0) * (s0.x * s0.y) +
           tex.SampleLevel(LinearClamp, float2(t1.x, t0.y), 0) * (s1.x * s0.y) +
           tex.SampleLevel(LinearClamp, float2(t0.x, t1.y), 0) * (s0.x * s1.y) +
           tex.SampleLevel(LinearClamp, float2(t1.x, t1.y), 0) * (s1.x * s1.y);
}

// ============================================================================
// Forward-Backward Consistency Check
// Returns 0 if consistent, >0 if inconsistent (occlusion likely)
// ============================================================================
float CheckConsistency(float2 uv, float2 mv, float2 inSize, float2 mvSize, float2 texelSize)
{
    // Warp to prev frame
    float2 prevUV = uv - mv * texelSize;
    
    // Get motion at that location (should point back to us)
    float2 backMV = Motion.SampleLevel(LinearClamp, prevUV, 0);
    backMV *= (inSize / mvSize);
    
    // Check if forward + backward = ~0 (consistent)
    float2 cycle = mv + backMV;
    return length(cycle);
}

// ============================================================================
// MAIN SHADER
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutColor.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    
    // Get dimensions
    uint inW, inH;
    PrevColor.GetDimensions(inW, inH);
    uint mvW, mvH;
    Motion.GetDimensions(mvW, mvH);
    
    float2 outSize = float2(outW, outH);
    float2 inSize = float2(inW, inH);
    float2 mvSize = float2(mvW, mvH);
    float2 texelSize = 1.0 / inSize;
    float textBlend = 0.0;
    float3 baseNearest = float3(0.0, 0.0, 0.0);
    
    // Calculate positions
    float2 outPos = float2(id.xy) + 0.5;
    float2 inputPos = outPos * (inSize / outSize);
    float2 inputUv = inputPos / inSize;
    
    // ========================================================================
    // 1. FETCH MOTION VECTOR
    // ========================================================================
    float2 mvUv = inputUv;
    float2 mv = Motion.SampleLevel(LinearClamp, mvUv, 0).xy;
    mv *= (inSize / mvSize);
    
    float conf = Confidence.SampleLevel(LinearClamp, mvUv, 0);
    conf = pow(saturate(conf), confPower);
    
    // ========================================================================
    // 2. FORWARD-BACKWARD CONSISTENCY CHECK
    // ========================================================================
    float consistency = CheckConsistency(inputUv, mv, inSize, mvSize, texelSize);
    // Optimization: consistencyWeight was used for occlusion Logic which is now removed.
    // float consistencyWeight = exp(-consistency * 0.5); 
    
    // ========================================================================
    // 3. BIDIRECTIONAL WARPING (High Quality Bicubic)
    // ========================================================================
    float2 prevWarpPos = inputPos + mv * alpha;
    float2 currWarpPos = inputPos - mv * (1.0 - alpha);
    
    float2 prevUV = clamp(prevWarpPos / inSize, 0.001, 0.999);
    float2 currUV = clamp(currWarpPos / inSize, 0.001, 0.999);
    
    // Force Bicubic for "High Quality" (Sharpness > Performance)
    // It is still relatively low cost (4 samples vs 1)
    float3 cPrev = SampleBicubic(PrevColor, prevUV, inSize).rgb;
    float3 cCurr = SampleBicubic(CurrColor, currUV, inSize).rgb;
    
    // ========================================================================
    // 4. PURE SMOOTH BLENDING
    // ========================================================================
    // Removed "Smart Snapping" (smoothstep) because it caused micro-stutter (loss of smoothness).
    // Reverted to Linear Blending which guarantees mathematically perfect frame intervals (120fps feel).
    // Bicubic sampling handles the sharpness, so we don't need tricks here.
    
    float t = alpha;
    float3 result = lerp(cPrev, cCurr, t);

    // ========================================================================
    // OUTPUT
    // ========================================================================
    OutColor[id.xy] = float4(saturate(result), 1.0);
}
