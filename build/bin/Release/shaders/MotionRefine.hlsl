// ============================================================================
// MOTION REFINEMENT v2 - Gradient-Descent Sub-pixel + Consistency Check
//
// Key improvements:
//   1. Uses coarse motion as initialization, then performs iterative
//      Lucas-Kanade gradient descent for sub-pixel accuracy
//   2. Forward-backward consistency check for occlusion detection
//   3. Adaptive convergence: high-confidence coarse vectors skip refinement
//   4. Tiny CNN-inspired channel attention (12->6->12 MLP-style gating)
// ============================================================================

Texture2D<float4>  CurrLuma       : register(t0);
Texture2D<float4>  PrevLuma       : register(t1);
Texture2D<float4>  CurrFeature2   : register(t2);
Texture2D<float4>  PrevFeature2   : register(t3);
Texture2D<float4>  CurrFeature3   : register(t4);
Texture2D<float4>  PrevFeature3   : register(t5);
Texture2D<float2> CoarseMotion   : register(t6);
Texture2D<float>  CoarseConf     : register(t7);
Texture2D<float2> BackwardMotion : register(t8);
Texture2D<float>  BackwardConf   : register(t9);
RWTexture2D<float2> MotionOut     : register(u0);
RWTexture2D<float>  ConfidenceOut : register(u1);
RWTexture2D<float4> AttnState1    : register(u2);
RWTexture2D<float4> AttnState2    : register(u3);
RWTexture2D<float4> AttnState3    : register(u4);

SamplerState LinearClamp : register(s0);

cbuffer RefineCB : register(b0) {
    int   radius;
    float motionScale;
    int   useBackward;
    float backwardScale;
    float attnLearnRate;
    float attnPriorMix;
    float attnStability;
    float _refinePad0;
};

// ============================================================================
// IFNet-Lite + FusionNet-Lite: Trainable MLP weights
//
// IFNet-Lite (MotionRefine): 12->8->16 MLP
//   Outputs: 12 attention gates + 2 flow residual + 1 occlusion + 1 quality
//
// FusionNet-Lite (Interpolate): 12->6->4 Synthesis MLP
//   Outputs: blend_weight + detail_factor + confidence_gate + sharpness
//
// Total: 128 trainable parameters
// ============================================================================
cbuffer AttentionWeightsCB : register(b1) {
    // === IFNet-Lite: 12->8->16 ===
    // Hidden weights: 8 units, each with 4 shared weights
    float4 mlpW_h0;
    float4 mlpW_h1;
    float4 mlpW_h2;
    float4 mlpW_h3;
    float4 mlpW_h4;
    float4 mlpW_h5;
    float4 mlpW_h6;
    float4 mlpW_h7;
    
    // Output weights: 4 shared vectors (8 hidden -> 16 output via grouping)
    float4 mlpW_out0;
    float4 mlpW_out1;
    float4 mlpW_out2;
    float4 mlpW_out3;
    
    // Hidden biases (8 values in 2 float4)
    float4 mlpBias_h0;   // h0, h1, h2, h3
    float4 mlpBias_h1;   // h4, h5, h6, h7
    
    // Output biases (16 values in 4 float4)
    float4 mlpBias_out0;  // attention gates 1-4
    float4 mlpBias_out1;  // attention gates 5-8
    float4 mlpBias_out2;  // attention gates 9-12
    float4 mlpBias_out3;  // residual_dx, residual_dy, occlusion, quality
    
    // Base attention weights for normalization fallback
    float4 baseW1;
    float4 baseW2;
    float4 baseW3;
    
    // === FusionNet-Lite: 12->6->4 (Synthesis MLP for Interpolate.hlsl) ===
    float4 synthW_h0;
    float4 synthW_h1;
    float4 synthW_h2;
    float4 synthW_h3;
    float4 synthW_h4;
    float4 synthW_h5;
    float4 synthW_out0;
    float4 synthW_out1;
    float4 synthBias_h0;  // h0..h3
    float4 synthBias_h1;  // h4, h5, pad, pad
    float4 synthBias_out; // 4 output biases
    
    // Control flag (0 = use hardcoded defaults, 1 = use trained weights)
    float useCustomWeights;
    float3 _pad;
};

#define PATCH_R 1

float SigmoidFast(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float4 NormalizeWeights(float4 w, float4 fallbackW) {
    float4 p = max(w, 0.0);
    float s = dot(p, 1.0);
    if (s <= 1e-6) return fallbackW;
    return p / s;
}

float4 BlendPrior(float4 baseW, float4 stateW, float mixAmount) {
    float4 mixed = lerp(baseW, stateW, saturate(mixAmount));
    return NormalizeWeights(mixed, baseW);
}

// Tiny channel-attention MLP:
//   Input  : 12 channels (3 x float4 energies)
//   Hidden : 6 units (ReLU)
//   Output : 12 gates (sigmoid), normalized to attention weights
//
// The weights here are static placeholders with sane defaults. You can replace
// them with offline-trained values later without changing call sites.
void CnnAttention12(
    float4 energy1,
    float4 energy2,
    float4 energy3,
    float4 baseW1,
    float4 baseW2,
    float4 baseW3,
    out float4 outW1,
    out float4 outW2,
    out float4 outW3) {

    float total = dot(energy1, 1.0) + dot(energy2, 1.0) + dot(energy3, 1.0) + 1e-6;
    float4 x1 = max(energy1 / total, 0.0);
    float4 x2 = max(energy2 / total, 0.0);
    float4 x3 = max(energy3 / total, 0.0);

    // Check if custom weights are enabled
    if (useCustomWeights > 0.5) {
        // IFNet-Lite: 12 -> 8 hidden (ReLU) with weight-sharing permutations
        float h0 = max(0.0, dot(x1, mlpW_h0.xyzw) + dot(x2, mlpW_h0.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h0.w, mlpW_h0.x, mlpW_h0.y, mlpW_h0.z)) + mlpBias_h0.x);
        float h1 = max(0.0, dot(x1, mlpW_h1.xyzw) + dot(x2, mlpW_h1.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h1.w, mlpW_h1.x, mlpW_h1.y, mlpW_h1.z)) + mlpBias_h0.y);
        float h2 = max(0.0, dot(x1, mlpW_h2.xyzw) + dot(x2, mlpW_h2.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h2.w, mlpW_h2.x, mlpW_h2.y, mlpW_h2.z)) + mlpBias_h0.z);
        float h3 = max(0.0, dot(x1, mlpW_h3.xyzw) + dot(x2, mlpW_h3.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h3.w, mlpW_h3.x, mlpW_h3.y, mlpW_h3.z)) + mlpBias_h0.w);
        float h4 = max(0.0, dot(x1, mlpW_h4.xyzw) + dot(x2, mlpW_h4.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h4.w, mlpW_h4.x, mlpW_h4.y, mlpW_h4.z)) + mlpBias_h1.x);
        float h5 = max(0.0, dot(x1, mlpW_h5.xyzw) + dot(x2, mlpW_h5.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h5.w, mlpW_h5.x, mlpW_h5.y, mlpW_h5.z)) + mlpBias_h1.y);
        float h6 = max(0.0, dot(x1, mlpW_h6.xyzw) + dot(x2, mlpW_h6.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h6.w, mlpW_h6.x, mlpW_h6.y, mlpW_h6.z)) + mlpBias_h1.z);
        float h7 = max(0.0, dot(x1, mlpW_h7.xyzw) + dot(x2, mlpW_h7.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h7.w, mlpW_h7.x, mlpW_h7.y, mlpW_h7.z)) + mlpBias_h1.w);
        
        // 8 -> 12 attention gates via 3 output groups (4 shared weight vectors)
        float4 z1 = saturate(mlpW_out0 * h0 + mlpW_out1 * h1 + mlpW_out2 * h2 + mlpW_out3 * h3 + mlpBias_out0);
        float4 z2 = saturate(mlpW_out0 * h4 + mlpW_out1 * h5 + mlpW_out2 * h6 + mlpW_out3 * h7 + mlpBias_out1);
        float4 z3 = saturate(mlpW_out0 * (h0+h4) + mlpW_out1 * (h1+h5) + mlpW_out2 * (h2+h6) + mlpW_out3 * (h3+h7) + mlpBias_out2);
        
        float4 g1 = z1;
        float4 g2 = z2;
        float4 g3 = z3;
        
        // Blend with priors
        float4 prior1 = baseW1 + 0.35 * x1;
        float4 prior2 = baseW2 + 0.35 * x2;
        float4 prior3 = baseW3 + 0.35 * x3;
        
        float4 w1 = g1 * prior1;
        float4 w2 = g2 * prior2;
        float4 w3 = g3 * prior3;
        
        float sumW = dot(w1, 1.0) + dot(w2, 1.0) + dot(w3, 1.0) + 1e-6;
        outW1 = w1 / sumW;
        outW2 = w2 / sumW;
        outW3 = w3 / sumW;
    } else {
        // 12 -> 6 (ReLU) - default weights
        float h0 = max(0.0, dot(x1, float4( 1.12, -0.31,  0.48,  0.86)) +
                            dot(x2, float4(-0.22,  0.71,  0.36, -0.17)) +
                            dot(x3, float4( 0.27,  0.19, -0.54,  0.42)) + 0.03);

        float h1 = max(0.0, dot(x1, float4(-0.49,  0.84,  0.24, -0.29)) +
                            dot(x2, float4( 0.93, -0.18,  0.41,  0.11)) +
                            dot(x3, float4(-0.15,  0.28,  0.63, -0.39)) - 0.01);

        float h2 = max(0.0, dot(x1, float4( 0.34,  0.27, -0.44,  0.75)) +
                            dot(x2, float4( 0.12, -0.66,  0.58,  0.47)) +
                            dot(x3, float4( 0.81, -0.25,  0.09, -0.14)) + 0.02);

        float h3 = max(0.0, dot(x1, float4( 0.58, -0.72,  0.18,  0.31)) +
                            dot(x2, float4(-0.41,  0.36,  0.77, -0.22)) +
                            dot(x3, float4( 0.24,  0.69, -0.17,  0.51)) + 0.04);

        float h4 = max(0.0, dot(x1, float4(-0.27,  0.41,  0.95, -0.33)) +
                            dot(x2, float4( 0.65,  0.23, -0.38,  0.54)) +
                            dot(x3, float4(-0.44,  0.16,  0.35,  0.72)) - 0.02);

        float h5 = max(0.0, dot(x1, float4( 0.73,  0.11, -0.29,  0.63)) +
                            dot(x2, float4( 0.08,  0.55,  0.22, -0.64)) +
                            dot(x3, float4( 0.47, -0.31,  0.84,  0.14)) + 0.01);

        // 6 -> 12 logits
        float4 z1;
        z1.x =  0.10 + 0.92*h0 - 0.28*h1 + 0.33*h2 + 0.19*h3 - 0.41*h4 + 0.27*h5;
        z1.y = -0.05 + 0.36*h0 + 0.74*h1 - 0.22*h2 + 0.15*h3 + 0.31*h4 - 0.18*h5;
        z1.z =  0.02 - 0.14*h0 + 0.42*h1 + 0.65*h2 - 0.27*h3 + 0.08*h4 + 0.24*h5;
        z1.w =  0.07 + 0.58*h0 + 0.11*h1 - 0.35*h2 + 0.66*h3 - 0.12*h4 + 0.21*h5;

        float4 z2;
        z2.x = -0.03 + 0.27*h0 - 0.16*h1 + 0.44*h2 + 0.31*h3 + 0.22*h4 - 0.37*h5;
        z2.y =  0.01 - 0.39*h0 + 0.53*h1 + 0.14*h2 + 0.29*h3 - 0.17*h4 + 0.48*h5;
        z2.z =  0.05 + 0.63*h0 + 0.24*h1 - 0.19*h2 - 0.33*h3 + 0.57*h4 + 0.09*h5;
        z2.w = -0.04 + 0.18*h0 + 0.37*h1 + 0.52*h2 - 0.21*h3 + 0.26*h4 - 0.11*h5;

        float4 z3;
        z3.x =  0.00 - 0.22*h0 + 0.45*h1 - 0.13*h2 + 0.71*h3 + 0.16*h4 + 0.28*h5;
        z3.y =  0.03 + 0.49*h0 - 0.24*h1 + 0.31*h2 + 0.08*h3 + 0.62*h4 - 0.29*h5;
        z3.z = -0.02 + 0.21*h0 + 0.18*h1 + 0.57*h2 + 0.26*h3 - 0.34*h4 + 0.41*h5;
        z3.w =  0.06 - 0.11*h0 + 0.67*h1 + 0.23*h2 - 0.16*h3 + 0.39*h4 + 0.12*h5;

        float scale = 1.35;
        float4 g1 = saturate(float4(
            SigmoidFast(z1.x * scale),
            SigmoidFast(z1.y * scale),
            SigmoidFast(z1.z * scale),
            SigmoidFast(z1.w * scale)));
        float4 g2 = saturate(float4(
            SigmoidFast(z2.x * scale),
            SigmoidFast(z2.y * scale),
            SigmoidFast(z2.z * scale),
            SigmoidFast(z2.w * scale)));
        float4 g3 = saturate(float4(
            SigmoidFast(z3.x * scale),
            SigmoidFast(z3.y * scale),
            SigmoidFast(z3.z * scale),
            SigmoidFast(z3.w * scale)));

        // Blend static priors with per-pixel energy evidence, then gate.
        float4 prior1 = baseW1 + 0.35 * x1;
        float4 prior2 = baseW2 + 0.35 * x2;
        float4 prior3 = baseW3 + 0.35 * x3;

        float4 w1 = g1 * prior1;
        float4 w2 = g2 * prior2;
        float4 w3 = g3 * prior3;

        float sumW = dot(w1, 1.0) + dot(w2, 1.0) + dot(w3, 1.0) + 1e-6;
        outW1 = w1 / sumW;
        outW2 = w2 / sumW;
        outW3 = w3 / sumW;
    }
}

// ============================================================================
// IFNetPostProcess - Learned motion residual + occlusion prediction
//
// Runs ONCE per pixel after LK converges. Uses the same hidden layer as
// CnnAttention12 but computes 4 extra outputs:
//   [0] motion residual X (tanh, ±0.5 px)
//   [1] motion residual Y (tanh, ±0.5 px)
//   [2] occlusion probability (sigmoid, 0=visible, 1=occluded)
//   [3] quality modifier (sigmoid, multiplied with confidence)
// ============================================================================
void IFNetPostProcess(
    float4 energy1,
    float4 energy2,
    float4 energy3,
    out float2 motionResidual,
    out float occlusion,
    out float qualityMod)
{
    if (useCustomWeights < 0.5) {
        motionResidual = float2(0, 0);
        occlusion = 0.0;
        qualityMod = 0.5;
        return;
    }
    
    float total = dot(energy1, 1.0) + dot(energy2, 1.0) + dot(energy3, 1.0) + 1e-6;
    float4 x1 = max(energy1 / total, 0.0);
    float4 x2 = max(energy2 / total, 0.0);
    float4 x3 = max(energy3 / total, 0.0);
    
    // Recompute 8 hidden units (same weight-sharing as CnnAttention12)
    float h0 = max(0.0, dot(x1, mlpW_h0.xyzw) + dot(x2, mlpW_h0.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h0.w, mlpW_h0.x, mlpW_h0.y, mlpW_h0.z)) + mlpBias_h0.x);
    float h1 = max(0.0, dot(x1, mlpW_h1.xyzw) + dot(x2, mlpW_h1.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h1.w, mlpW_h1.x, mlpW_h1.y, mlpW_h1.z)) + mlpBias_h0.y);
    float h2 = max(0.0, dot(x1, mlpW_h2.xyzw) + dot(x2, mlpW_h2.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h2.w, mlpW_h2.x, mlpW_h2.y, mlpW_h2.z)) + mlpBias_h0.z);
    float h3 = max(0.0, dot(x1, mlpW_h3.xyzw) + dot(x2, mlpW_h3.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h3.w, mlpW_h3.x, mlpW_h3.y, mlpW_h3.z)) + mlpBias_h0.w);
    float h4 = max(0.0, dot(x1, mlpW_h4.xyzw) + dot(x2, mlpW_h4.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h4.w, mlpW_h4.x, mlpW_h4.y, mlpW_h4.z)) + mlpBias_h1.x);
    float h5 = max(0.0, dot(x1, mlpW_h5.xyzw) + dot(x2, mlpW_h5.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h5.w, mlpW_h5.x, mlpW_h5.y, mlpW_h5.z)) + mlpBias_h1.y);
    float h6 = max(0.0, dot(x1, mlpW_h6.xyzw) + dot(x2, mlpW_h6.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h6.w, mlpW_h6.x, mlpW_h6.y, mlpW_h6.z)) + mlpBias_h1.z);
    float h7 = max(0.0, dot(x1, mlpW_h7.xyzw) + dot(x2, mlpW_h7.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(mlpW_h7.w, mlpW_h7.x, mlpW_h7.y, mlpW_h7.z)) + mlpBias_h1.w);
    
    // Extra outputs: antisymmetric diff combinations for flow/occlusion
    float4 extra = mlpW_out0 * (h0-h4) + mlpW_out1 * (h1-h5) + mlpW_out2 * (h2-h6) + mlpW_out3 * (h3-h7) + mlpBias_out3;
    
    // Motion residual: tanh activation, scaled to ±0.5 pixels
    motionResidual = float2(tanh(extra.x), tanh(extra.y)) * 0.5;
    
    // Occlusion: sigmoid (0=fully visible, 1=fully occluded)
    occlusion = SigmoidFast(extra.z);
    
    // Quality modifier: sigmoid (multiplied with confidence)
    qualityMod = SigmoidFast(extra.w);
}

// Compute image gradient at fractional position via bilinear sampling
float4 SampleLuma(float2 pos, float2 invSize) {
    float2 uv = clamp(pos * invSize, 0.0, 0.999);
    return PrevLuma.SampleLevel(LinearClamp, uv, 0);
}

float4 SampleFeature2(float2 pos, float2 invSize) {
    float2 uv = clamp(pos * invSize, 0.0, 0.999);
    return PrevFeature2.SampleLevel(LinearClamp, uv, 0);
}

float4 SampleFeature3(float2 pos, float2 invSize) {
    float2 uv = clamp(pos * invSize, 0.0, 0.999);
    return PrevFeature3.SampleLevel(LinearClamp, uv, 0);
}

float4 SampleCurrLuma(float2 pos, float2 invSize) {
    float2 uv = clamp(pos * invSize, 0.0, 0.999);
    return CurrLuma.SampleLevel(LinearClamp, uv, 0);
}

float4 SampleCurrFeature2(float2 pos, float2 invSize) {
    float2 uv = clamp(pos * invSize, 0.0, 0.999);
    return CurrFeature2.SampleLevel(LinearClamp, uv, 0);
}

float4 SampleCurrFeature3(float2 pos, float2 invSize) {
    float2 uv = clamp(pos * invSize, 0.0, 0.999);
    return CurrFeature3.SampleLevel(LinearClamp, uv, 0);
}

// Compute ZNCC for quality measurement (Multi-channel)
float EvalZNCC(
    int2 pos,
    float2 mv,
    uint w,
    uint h,
    float2 invSize,
    float4 priorW1,
    float4 priorW2,
    float4 priorW3) {
    int2 maxPos = int2(int(w) - 1, int(h) - 1);
    int n = 0;
    float4 sumC = 0, sumP = 0;
    float4 sumC2 = 0, sumP2 = 0;
    float4 sumC3 = 0, sumP3 = 0;

    [loop] for (int by = -PATCH_R; by <= PATCH_R; ++by) {
        [loop] for (int bx = -PATCH_R; bx <= PATCH_R; ++bx) {
            int2 cPos = clamp(pos + int2(bx, by), int2(0, 0), maxPos);
            float4 cVal = CurrLuma.Load(int3(cPos, 0));
            float4 cVal2 = CurrFeature2.Load(int3(cPos, 0));
            float4 cVal3 = CurrFeature3.Load(int3(cPos, 0));
            float4 pVal = SampleLuma(float2(cPos) + 0.5 + mv, invSize);
            float4 pVal2 = SampleFeature2(float2(cPos) + 0.5 + mv, invSize);
            float4 pVal3 = SampleFeature3(float2(cPos) + 0.5 + mv, invSize);
            sumC += cVal; sumP += pVal;
            sumC2 += cVal2; sumP2 += pVal2;
            sumC3 += cVal3; sumP3 += pVal3;
            n++;
        }
    }
    float4 meanC = sumC / float(n);
    float4 meanP = sumP / float(n);
    float4 meanC2 = sumC2 / float(n);
    float4 meanP2 = sumP2 / float(n);
    float4 meanC3 = sumC3 / float(n);
    float4 meanP3 = sumP3 / float(n);

    float4 cc = 0, varC = 0, varP = 0;
    float4 cc2 = 0, varC2 = 0, varP2 = 0;
    float4 cc3 = 0, varC3 = 0, varP3 = 0;
    [loop] for (int by2 = -PATCH_R; by2 <= PATCH_R; ++by2) {
        [loop] for (int bx2 = -PATCH_R; bx2 <= PATCH_R; ++bx2) {
            int2 cPos = clamp(pos + int2(bx2, by2), int2(0, 0), maxPos);
            float4 cVal = CurrLuma.Load(int3(cPos, 0)) - meanC;
            float4 cVal2 = CurrFeature2.Load(int3(cPos, 0)) - meanC2;
            float4 cVal3 = CurrFeature3.Load(int3(cPos, 0)) - meanC3;
            float4 pVal = SampleLuma(float2(cPos) + 0.5 + mv, invSize) - meanP;
            float4 pVal2 = SampleFeature2(float2(cPos) + 0.5 + mv, invSize) - meanP2;
            float4 pVal3 = SampleFeature3(float2(cPos) + 0.5 + mv, invSize) - meanP3;
            cc   += cVal * pVal;
            varC += cVal * cVal;
            varP += pVal * pVal;
            cc2  += cVal2 * pVal2;
            varC2 += cVal2 * cVal2;
            varP2 += pVal2 * pVal2;
            cc3  += cVal3 * pVal3;
            varC3 += cVal3 * cVal3;
            varP3 += pVal3 * pVal3;
        }
    }
    float4 denom = sqrt(max(varC, 1e-8) * max(varP, 1e-8));
    float4 zncc4 = cc / denom;
    
    float4 denom2 = sqrt(max(varC2, 1e-8) * max(varP2, 1e-8));
    float4 zncc4_2 = cc2 / denom2;

    float4 denom3 = sqrt(max(varC3, 1e-8) * max(varP3, 1e-8));
    float4 zncc4_3 = cc3 / denom3;
    
    // CNN-inspired channel attention for ZNCC aggregation.
    float4 dynamicWeights1, dynamicWeights2, dynamicWeights3;
    CnnAttention12(
        max(varC, 0.0),
        max(varC2, 0.0),
        max(varC3, 0.0),
        priorW1,
        priorW2,
        priorW3,
        dynamicWeights1,
        dynamicWeights2,
        dynamicWeights3);
    
    return dot(zncc4, dynamicWeights1) + dot(zncc4_2, dynamicWeights2) + dot(zncc4_3, dynamicWeights3);
}

// Lucas-Kanade iterative refinement step (Multi-channel)
// Computes gradient of the error and takes a step to minimize it
float2 LKStep(
    int2 pos,
    float2 mv,
    uint w,
    uint h,
    float2 invSize,
    float4 priorW1,
    float4 priorW2,
    float4 priorW3) {
    int2 maxPos = int2(int(w) - 1, int(h) - 1);
    
    // Build the 2x2 structure tensor and the mismatch vector
    float A00 = 0, A01 = 0, A11 = 0;
    float b0 = 0, b1 = 0;

    [loop] for (int by = -PATCH_R; by <= PATCH_R; ++by) {
        [loop] for (int bx = -PATCH_R; bx <= PATCH_R; ++bx) {
            int2 cPos = clamp(pos + int2(bx, by), int2(0, 0), maxPos);
            float2 pPos = float2(cPos) + 0.5 + mv;
            
            // Spatial gradients in prev frame at warped position (Multi-channel)
            float4 Ix = (SampleLuma(pPos + float2(1, 0), invSize) - SampleLuma(pPos + float2(-1, 0), invSize)) * 0.5;
            float4 Iy = (SampleLuma(pPos + float2(0, 1), invSize) - SampleLuma(pPos + float2(0, -1), invSize)) * 0.5;
            
            float4 Ix2 = (SampleFeature2(pPos + float2(1, 0), invSize) - SampleFeature2(pPos + float2(-1, 0), invSize)) * 0.5;
            float4 Iy2 = (SampleFeature2(pPos + float2(0, 1), invSize) - SampleFeature2(pPos + float2(0, -1), invSize)) * 0.5;

            float4 Ix3 = (SampleFeature3(pPos + float2(1, 0), invSize) - SampleFeature3(pPos + float2(-1, 0), invSize)) * 0.5;
            float4 Iy3 = (SampleFeature3(pPos + float2(0, 1), invSize) - SampleFeature3(pPos + float2(0, -1), invSize)) * 0.5;
            
            // Temporal difference
            float4 It = SampleLuma(pPos, invSize) - CurrLuma.Load(int3(cPos, 0));
            float4 It2 = SampleFeature2(pPos, invSize) - CurrFeature2.Load(int3(cPos, 0));
            float4 It3 = SampleFeature3(pPos, invSize) - CurrFeature3.Load(int3(cPos, 0));
            
            // Gaussian weight (center-weighted patch)
            float dist2 = float(bx * bx + by * by);
            float gw = exp(-dist2 / 3.0);
            
            // CNN-inspired channel attention for LK tensor accumulation.
            float4 gradEnergy1 = Ix * Ix + Iy * Iy;
            float4 gradEnergy2 = Ix2 * Ix2 + Iy2 * Iy2;
            float4 gradEnergy3 = Ix3 * Ix3 + Iy3 * Iy3;
            
            float4 cw1, cw2, cw3;
            CnnAttention12(
                max(gradEnergy1, 0.0),
                max(gradEnergy2, 0.0),
                max(gradEnergy3, 0.0),
                priorW1,
                priorW2,
                priorW3,
                cw1,
                cw2,
                cw3);
            
            // Accumulate structure tensor over all channels
            A00 += (dot(Ix * Ix * cw1, 1.0) + dot(Ix2 * Ix2 * cw2, 1.0) + dot(Ix3 * Ix3 * cw3, 1.0)) * gw;
            A01 += (dot(Ix * Iy * cw1, 1.0) + dot(Ix2 * Iy2 * cw2, 1.0) + dot(Ix3 * Iy3 * cw3, 1.0)) * gw;
            A11 += (dot(Iy * Iy * cw1, 1.0) + dot(Iy2 * Iy2 * cw2, 1.0) + dot(Iy3 * Iy3 * cw3, 1.0)) * gw;
            b0  -= (dot(Ix * It * cw1, 1.0) + dot(Ix2 * It2 * cw2, 1.0) + dot(Ix3 * It3 * cw3, 1.0)) * gw;
            b1  -= (dot(Iy * It * cw1, 1.0) + dot(Iy2 * It2 * cw2, 1.0) + dot(Iy3 * It3 * cw3, 1.0)) * gw;
        }
    }

    // Solve 2x2 system: A * delta = b
    float det = A00 * A11 - A01 * A01;
    if (abs(det) < 1e-6) return float2(0, 0);
    
    float invDet = 1.0 / det;
    float2 delta;
    delta.x = (A11 * b0 - A01 * b1) * invDet;
    delta.y = (A00 * b1 - A01 * b0) * invDet;
    
    // Clamp step size to prevent divergence
    float stepLen = length(delta);
    if (stepLen > 2.0) delta *= 2.0 / stepLen;
    
    return delta;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    CurrLuma.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int2 pos = int2(id.xy);
    float2 invSize = 1.0 / float2(w, h);
    float2 uv = (float2(pos) + 0.5) * invSize;

    // Read coarse motion and upscale
    float2 coarseMV = CoarseMotion.SampleLevel(LinearClamp, uv, 0).xy * motionScale;
    float coarseConf = saturate(CoarseConf.SampleLevel(LinearClamp, uv, 0));

    float4 kBaseW1 = float4(0.15, 0.1, 0.1, 0.2);
    float4 kBaseW2 = float4(0.1, 0.1, 0.15, 0.1);
    float4 kBaseW3 = float4(0.1, 0.1, 0.1, 0.1);

    float4 stateW1 = NormalizeWeights(AttnState1[pos], kBaseW1);
    float4 stateW2 = NormalizeWeights(AttnState2[pos], kBaseW2);
    float4 stateW3 = NormalizeWeights(AttnState3[pos], kBaseW3);

    float4 priorW1 = BlendPrior(kBaseW1, stateW1, attnPriorMix);
    float4 priorW2 = BlendPrior(kBaseW2, stateW2, attnPriorMix);
    float4 priorW3 = BlendPrior(kBaseW3, stateW3, attnPriorMix);

    // Fast path: very high confidence near-zero motion doesn't need refinement
    if (coarseConf > 0.95 && dot(coarseMV, coarseMV) < 0.04) {
        MotionOut[id.xy] = coarseMV;
        ConfidenceOut[id.xy] = coarseConf;
        AttnState1[id.xy] = stateW1;
        AttnState2[id.xy] = stateW2;
        AttnState3[id.xy] = stateW3;
        return;
    }

    // --- 1-pixel search around coarse ---
    float znccCoarse = EvalZNCC(pos, coarseMV, w, h, invSize, priorW1, priorW2, priorW3);
    float2 bestCoarseMV = coarseMV;
    [loop] for (int dy = -1; dy <= 1; ++dy) {
        [loop] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float2 testMV = coarseMV + float2(dx, dy);
            float z = EvalZNCC(pos, testMV, w, h, invSize, priorW1, priorW2, priorW3);
            if (z > znccCoarse) {
                znccCoarse = z;
                bestCoarseMV = testMV;
            }
        }
    }
    coarseMV = bestCoarseMV;

    // --- Iterative Lucas-Kanade refinement ---
    float2 mv = coarseMV;
    int maxIter = clamp(radius, 1, 4);
    
    // Reduce iterations for high-confidence inputs (but keep more for quality)
    if (coarseConf > 0.85) maxIter = 2;
    else if (coarseConf > 0.6) maxIter = min(maxIter, 3);
    else if (coarseConf > 0.4) maxIter = min(maxIter, 4);

    [loop] for (int iter = 0; iter < maxIter; ++iter) {
        float2 delta = LKStep(pos, mv, w, h, invSize, priorW1, priorW2, priorW3);
        mv += delta;
        // Converged if step is very small
        if (dot(delta, delta) < 0.0001) break;
    }

    // Keep refined motion within reasonable range of coarse
    float maxDrift = float(radius) + 1.0;
    float2 drift = mv - coarseMV;
    if (length(drift) > maxDrift) {
        mv = coarseMV + normalize(drift) * maxDrift;
    }

    // --- Quality evaluation and fallback ---
    float znccRefined = EvalZNCC(pos, mv, w, h, invSize, priorW1, priorW2, priorW3);
    
    if (znccCoarse > znccRefined) {
        mv = coarseMV;
        znccRefined = znccCoarse;
    }

    // --- Forward-backward consistency check ---
    float consistency = 1.0;
    if (useBackward != 0) {
        float2 matchPos = float2(pos) + 0.5 + mv;
        float2 matchUv = clamp(matchPos * invSize, 0.0, 0.999);
        float2 backMV = BackwardMotion.SampleLevel(LinearClamp, matchUv, 0).xy * backwardScale;
        float backConf = saturate(BackwardConf.SampleLevel(LinearClamp, matchUv, 0));
        
        float fbError = length(mv + backMV);
        consistency = exp(-fbError * fbError / 8.0) * lerp(0.5, 1.0, backConf);
    }

    // --- Quality evaluation ---
    float matchQuality = saturate((znccRefined + 1.0) * 0.5);
    
    float confidence = matchQuality * consistency;
    confidence = lerp(confidence, coarseConf, 0.25); // Blend with coarse for stability
    confidence = clamp(confidence, 0.03, 0.99);

    // --- WHT Periodicity-Guided Motion Selection ---
    // High periodicity = repetitive texture = multiple similar motion candidates
    // Use periodicity to prefer motion vectors that are consistent with neighbors
    // instead of reducing confidence (we WANT to interpolate textures)
    float periodicity = CurrFeature3.Load(int3(pos, 0)).w;
    
    // For periodic textures, add spatial consistency bonus
    // If nearby pixels have similar motion, boost confidence
    if (periodicity > 0.3) {
        float2 neighborMV = MotionOut[int2(pos) + int2(1, 0)];
        float2 neighborMV2 = MotionOut[int2(pos) + int2(-1, 0)];
        float2 neighborMV3 = MotionOut[int2(pos) + int2(0, 1)];
        float2 neighborMV4 = MotionOut[int2(pos) + int2(0, -1)];
        float neighborConsistency = 1.0 - length(mv - neighborMV) * 0.5;
        neighborConsistency = max(neighborConsistency, 1.0 - length(mv - neighborMV2) * 0.5);
        neighborConsistency = max(neighborConsistency, 1.0 - length(mv - neighborMV3) * 0.5);
        neighborConsistency = max(neighborConsistency, 1.0 - length(mv - neighborMV4) * 0.5);
        neighborConsistency = saturate(neighborConsistency);
        
        // For periodic textures, require spatial consistency
        confidence *= lerp(1.0, neighborConsistency, periodicity * 0.5);
    }

    // --- Online attention adaptation ---
    // Estimate local channel reliability at the final refined motion and update
    // temporal attention priors with a bounded EMA step.
    float2 pPos = float2(pos) + 0.5 + mv;
    float4 Ix = (SampleLuma(pPos + float2(1, 0), invSize) - SampleLuma(pPos + float2(-1, 0), invSize)) * 0.5;
    float4 Iy = (SampleLuma(pPos + float2(0, 1), invSize) - SampleLuma(pPos + float2(0, -1), invSize)) * 0.5;
    float4 Ix2 = (SampleFeature2(pPos + float2(1, 0), invSize) - SampleFeature2(pPos + float2(-1, 0), invSize)) * 0.5;
    float4 Iy2 = (SampleFeature2(pPos + float2(0, 1), invSize) - SampleFeature2(pPos + float2(0, -1), invSize)) * 0.5;
    float4 Ix3 = (SampleFeature3(pPos + float2(1, 0), invSize) - SampleFeature3(pPos + float2(-1, 0), invSize)) * 0.5;
    float4 Iy3 = (SampleFeature3(pPos + float2(0, 1), invSize) - SampleFeature3(pPos + float2(0, -1), invSize)) * 0.5;

    float4 gradEnergy1 = max(Ix * Ix + Iy * Iy, 0.0);
    float4 gradEnergy2 = max(Ix2 * Ix2 + Iy2 * Iy2, 0.0);
    float4 gradEnergy3 = max(Ix3 * Ix3 + Iy3 * Iy3, 0.0);

    // --- IFNet-Lite: Learned motion residual + occlusion (like RIFE's IFNet) ---
    float2 mlpResidual;
    float mlpOcclusion, mlpQuality;
    IFNetPostProcess(gradEnergy1, gradEnergy2, gradEnergy3,
                     mlpResidual, mlpOcclusion, mlpQuality);
    
    // Apply learned sub-pixel correction (improves on LK classical result)
    mv += mlpResidual;
    
    // Modulate confidence with learned quality and occlusion
    confidence *= lerp(0.5, 1.0, mlpQuality);           // Quality gate
    confidence *= (1.0 - mlpOcclusion * 0.7);            // Reduce in occluded regions
    confidence = clamp(confidence, 0.03, 0.99);

    float4 targetW1, targetW2, targetW3;
    CnnAttention12(
        gradEnergy1,
        gradEnergy2,
        gradEnergy3,
        priorW1,
        priorW2,
        priorW3,
        targetW1,
        targetW2,
        targetW3);

    float learn = saturate(attnLearnRate) * (0.35 + 0.65 * confidence);
    float stability = saturate(attnStability);
    float effectiveLearn = learn * lerp(1.0, 0.25, stability);

    float4 deltaW1 = clamp(targetW1 - stateW1, -0.08, 0.08);
    float4 deltaW2 = clamp(targetW2 - stateW2, -0.08, 0.08);
    float4 deltaW3 = clamp(targetW3 - stateW3, -0.08, 0.08);

    float4 nextW1 = NormalizeWeights(stateW1 + deltaW1 * effectiveLearn, kBaseW1);
    float4 nextW2 = NormalizeWeights(stateW2 + deltaW2 * effectiveLearn, kBaseW2);
    float4 nextW3 = NormalizeWeights(stateW3 + deltaW3 * effectiveLearn, kBaseW3);

    MotionOut[id.xy] = mv;
    ConfidenceOut[id.xy] = confidence;
    AttnState1[id.xy] = nextW1;
    AttnState2[id.xy] = nextW2;
    AttnState3[id.xy] = nextW3;
}
