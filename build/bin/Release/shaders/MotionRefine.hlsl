// ============================================================================
// MOTION REFINEMENT v2 - Gradient-Descent Sub-pixel + Consistency Check
//
// Key improvements:
//   1. Uses coarse motion as initialization, then performs iterative
//      Lucas-Kanade gradient descent for sub-pixel accuracy
//   2. Forward-backward consistency check for occlusion detection
//   3. Adaptive convergence: high-confidence coarse vectors skip refinement
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

SamplerState LinearClamp : register(s0);

cbuffer RefineCB : register(b0) {
    int   radius;
    float motionScale;
    int   useBackward;
    float backwardScale;
};

#define PATCH_R 1

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
float EvalZNCC(int2 pos, float2 mv, uint w, uint h, float2 invSize) {
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
    
    // Advanced CNN: Feature-wise Self-Attention (12 channels)
    float totalVar = varC.x + varC.y + varC.z + varC.w + varC2.x + varC2.y + varC2.z + varC2.w + varC3.x + varC3.y + varC3.z + varC3.w + 1e-5;
    float4 attention1 = varC / totalVar;
    float4 attention2 = varC2 / totalVar;
    float4 attention3 = varC3 / totalVar;
    
    float4 baseW1 = float4(0.15, 0.1, 0.1, 0.2);
    float4 baseW2 = float4(0.1, 0.1, 0.15, 0.1);
    float4 baseW3 = float4(0.1, 0.1, 0.1, 0.1);
    
    float4 dynamicWeights1 = lerp(baseW1, attention1, 0.85);
    float4 dynamicWeights2 = lerp(baseW2, attention2, 0.85);
    float4 dynamicWeights3 = lerp(baseW3, attention3, 0.85);
    
    float sumW = dot(dynamicWeights1, 1.0) + dot(dynamicWeights2, 1.0) + dot(dynamicWeights3, 1.0);
    dynamicWeights1 /= sumW;
    dynamicWeights2 /= sumW;
    dynamicWeights3 /= sumW;
    
    return dot(zncc4, dynamicWeights1) + dot(zncc4_2, dynamicWeights2) + dot(zncc4_3, dynamicWeights3);
}

// Lucas-Kanade iterative refinement step (Multi-channel)
// Computes gradient of the error and takes a step to minimize it
float2 LKStep(int2 pos, float2 mv, uint w, uint h, float2 invSize) {
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
            
            // Advanced CNN: Spatial-Channel Attention for Lucas-Kanade (12 channels)
            // Focus on channels with the strongest gradients at this specific pixel
            float4 gradEnergy1 = Ix * Ix + Iy * Iy;
            float4 gradEnergy2 = Ix2 * Ix2 + Iy2 * Iy2;
            float4 gradEnergy3 = Ix3 * Ix3 + Iy3 * Iy3;
            float totalEnergy = gradEnergy1.x + gradEnergy1.y + gradEnergy1.z + gradEnergy1.w + gradEnergy2.x + gradEnergy2.y + gradEnergy2.z + gradEnergy2.w + gradEnergy3.x + gradEnergy3.y + gradEnergy3.z + gradEnergy3.w + 1e-5;
            float4 attention1 = gradEnergy1 / totalEnergy;
            float4 attention2 = gradEnergy2 / totalEnergy;
            float4 attention3 = gradEnergy3 / totalEnergy;
            
            float4 baseW1 = float4(0.15, 0.1, 0.1, 0.2);
            float4 baseW2 = float4(0.1, 0.1, 0.15, 0.1);
            float4 baseW3 = float4(0.1, 0.1, 0.1, 0.1);
            
            float4 cw1 = lerp(baseW1, attention1, 0.8);
            float4 cw2 = lerp(baseW2, attention2, 0.8);
            float4 cw3 = lerp(baseW3, attention3, 0.8);
            
            float sumW = dot(cw1, 1.0) + dot(cw2, 1.0) + dot(cw3, 1.0);
            cw1 /= sumW;
            cw2 /= sumW;
            cw3 /= sumW;
            
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

    // Fast path: very high confidence near-zero motion doesn't need refinement
    if (coarseConf > 0.95 && dot(coarseMV, coarseMV) < 0.04) {
        MotionOut[id.xy] = coarseMV;
        ConfidenceOut[id.xy] = coarseConf;
        return;
    }

    // --- 1-pixel search around coarse ---
    float znccCoarse = EvalZNCC(pos, coarseMV, w, h, invSize);
    float2 bestCoarseMV = coarseMV;
    [loop] for (int dy = -1; dy <= 1; ++dy) {
        [loop] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float2 testMV = coarseMV + float2(dx, dy);
            float z = EvalZNCC(pos, testMV, w, h, invSize);
            if (z > znccCoarse) {
                znccCoarse = z;
                bestCoarseMV = testMV;
            }
        }
    }
    coarseMV = bestCoarseMV;

    // --- Iterative Lucas-Kanade refinement ---
    float2 mv = coarseMV;
    int maxIter = clamp(radius, 1, 3);
    
    // Reduce iterations for high-confidence inputs
    if (coarseConf > 0.75) maxIter = 1;
    else if (coarseConf > 0.5) maxIter = min(maxIter, 2);

    [loop] for (int iter = 0; iter < maxIter; ++iter) {
        float2 delta = LKStep(pos, mv, w, h, invSize);
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
    float znccRefined = EvalZNCC(pos, mv, w, h, invSize);
    
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

    MotionOut[id.xy] = mv;
    ConfidenceOut[id.xy] = confidence;
}
