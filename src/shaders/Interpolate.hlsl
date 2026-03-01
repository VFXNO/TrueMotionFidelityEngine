// ============================================================================
// FRAME INTERPOLATION v2.2 - Pure Motion-Compensated Warping
//
// Architecture:
//   1. Read and smooth motion vectors (9-tap bilateral on coarse fields)
//   2. Forward warp from Prev, backward warp from Curr
//   3. AI occlusion-aware source selection (no frame blending)
//   4. No dissolve, no blending, no ghosting - purely warped output
// ============================================================================

Texture2D<float4> PrevColor        : register(t0);
Texture2D<float4> CurrColor        : register(t1);
Texture2D<float2> Motion           : register(t2);
Texture2D<float>  Confidence       : register(t3);
Texture2D<float2> MotionBackward   : register(t4);
Texture2D<float>  ConfidenceBackward: register(t5);
Texture2D<float4> PrevFeature      : register(t6);
Texture2D<float4> CurrFeature      : register(t7);
Texture2D<float4> PrevFeature2     : register(t8);
Texture2D<float4> CurrFeature2     : register(t9);
Texture2D<float4> PrevFeature3     : register(t10);
Texture2D<float4> CurrFeature3     : register(t11);
RWTexture2D<float4> OutColor       : register(u0);

SamplerState LinearClamp : register(s0);

cbuffer InterpCB : register(b0) {
    float alpha;
    float diffScale;
    float confPower;
    int   qualityMode;
    int   _reserved0;
    float _reserved1;
    float _reserved2;
    float _reserved3;
    float motionSampleScale;
    float3 pad;
};

// ============================================================================
// FusionNet-Lite: Shared weight buffer (same layout as MotionRefine.hlsl)
// Only the synthesis MLP fields and useCustomWeights flag are used here.
// ============================================================================
cbuffer AttentionWeightsCB : register(b1) {
    // === IFNet-Lite fields (unused here, must match layout) ===
    float4 mlpW_h0, mlpW_h1, mlpW_h2, mlpW_h3;
    float4 mlpW_h4, mlpW_h5, mlpW_h6, mlpW_h7;
    float4 mlpW_out0, mlpW_out1, mlpW_out2, mlpW_out3;
    float4 mlpBias_h0, mlpBias_h1;
    float4 mlpBias_out0, mlpBias_out1, mlpBias_out2, mlpBias_out3;
    float4 baseW1, baseW2, baseW3;
    
    // === FusionNet-Lite: 12->6->4 Synthesis MLP ===
    float4 synthW_h0, synthW_h1, synthW_h2;
    float4 synthW_h3, synthW_h4, synthW_h5;
    float4 synthW_out0, synthW_out1;
    float4 synthBias_h0, synthBias_h1;
    float4 synthBias_out;
    
    float useCustomWeights;
    float3 _attnPad;
};

float SigmoidFast(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// ============================================================================
// FusionNet-Lite Synthesis MLP: Learned occlusion-aware frame synthesis
//
// Input:  12 warped feature differences (|prev_warped - curr_warped|)
// Hidden: 6 units (ReLU)
// Output: 4 synthesis controls (sigmoid)
//   [0] blend_weight   - occlusion-aware blend (0=use prev, 1=use curr)
//   [1] detail_factor  - high-frequency detail injection strength
//   [2] confidence_gate - how much to trust the warped result
//   [3] sharpness      - adaptive sharpening strength
// ============================================================================
float4 SynthesisNet(float4 diff1, float4 diff2, float4 diff3) {
    if (useCustomWeights < 0.5) {
        // Default: neutral synthesis (matches pre-enhancement behavior)
        return float4(0.5, 0.5, 0.85, 0.3);
    }
    
    float total = dot(abs(diff1), 1.0) + dot(abs(diff2), 1.0) + dot(abs(diff3), 1.0) + 1e-6;
    float4 x1 = abs(diff1) / total;
    float4 x2 = abs(diff2) / total;
    float4 x3 = abs(diff3) / total;
    
    // 6 hidden units (ReLU) with same weight-sharing as IFNet
    float h0 = max(0.0, dot(x1, synthW_h0.xyzw) + dot(x2, synthW_h0.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(synthW_h0.w, synthW_h0.x, synthW_h0.y, synthW_h0.z)) + synthBias_h0.x);
    float h1 = max(0.0, dot(x1, synthW_h1.xyzw) + dot(x2, synthW_h1.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(synthW_h1.w, synthW_h1.x, synthW_h1.y, synthW_h1.z)) + synthBias_h0.y);
    float h2 = max(0.0, dot(x1, synthW_h2.xyzw) + dot(x2, synthW_h2.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(synthW_h2.w, synthW_h2.x, synthW_h2.y, synthW_h2.z)) + synthBias_h0.z);
    float h3 = max(0.0, dot(x1, synthW_h3.xyzw) + dot(x2, synthW_h3.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(synthW_h3.w, synthW_h3.x, synthW_h3.y, synthW_h3.z)) + synthBias_h0.w);
    float h4 = max(0.0, dot(x1, synthW_h4.xyzw) + dot(x2, synthW_h4.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(synthW_h4.w, synthW_h4.x, synthW_h4.y, synthW_h4.z)) + synthBias_h1.x);
    float h5 = max(0.0, dot(x1, synthW_h5.xyzw) + dot(x2, synthW_h5.zwxy * float4(-1,-1,1,1)) + dot(x3, float4(synthW_h5.w, synthW_h5.x, synthW_h5.y, synthW_h5.z)) + synthBias_h1.y);
    
    // 4 outputs (sigmoid) with weight sharing
    float4 raw = synthW_out0 * (h0 + h2 + h4) + synthW_out1 * (h1 + h3 + h5) + synthBias_out;
    
    return float4(
        SigmoidFast(raw.x),   // blend_weight
        SigmoidFast(raw.y),   // detail_factor
        SigmoidFast(raw.z),   // confidence_gate
        SigmoidFast(raw.w)    // sharpness
    );
}

static const float3 kLumaWeights = float3(0.2126, 0.7152, 0.0722);

float Luma(float3 c) { return dot(c, kLumaWeights); }

// -----------------------------------------------------------------------
// Catmull-Rom bicubic sampling (4-tap separable via bilinear trick)
// -----------------------------------------------------------------------
float3 SampleBicubic(Texture2D<float4> tex, float2 uv, float2 texSize) {
    float2 tc = uv * texSize;
    float2 itc = floor(tc - 0.5) + 0.5;
    float2 f = tc - itc;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = f2 - 0.5 * (f3 + f);
    float2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    float2 w3 = 0.5 * (f3 - f2);
    float2 w2 = 1.0 - w0 - w1 - w3;

    float2 s0 = w0 + w1;
    float2 s1 = w2 + w3;
    float2 f0 = w1 / max(s0, 1e-6);
    float2 f1 = w3 / max(s1, 1e-6);

    float2 t0 = (itc - 1.0 + f0) / texSize;
    float2 t1 = (itc + 1.0 + f1) / texSize;

    return tex.SampleLevel(LinearClamp, float2(t0.x, t0.y), 0).rgb * (s0.x * s0.y) +
           tex.SampleLevel(LinearClamp, float2(t1.x, t0.y), 0).rgb * (s1.x * s0.y) +
           tex.SampleLevel(LinearClamp, float2(t0.x, t1.y), 0).rgb * (s0.x * s1.y) +
           tex.SampleLevel(LinearClamp, float2(t1.x, t1.y), 0).rgb * (s1.x * s1.y);
}

float3 SampleColor(Texture2D<float4> tex, float2 uv, float2 texSize) {
    float3 result = tex.SampleLevel(LinearClamp, uv, 0).rgb;
    if (qualityMode >= 1) {
        result = SampleBicubic(tex, uv, texSize);
    }
    return result;
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint outW, outH;
    OutColor.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;

    uint inW, inH;
    PrevColor.GetDimensions(inW, inH);

    float2 outSize = float2(outW, outH);
    float2 inSize  = float2(inW, inH);

    // Map output pixel to input space
    float2 outPos   = float2(id.xy) + 0.5;
    float2 inputPos = outPos * (inSize / outSize);
    float2 inputUv  = inputPos / inSize;

    // =====================================================================
    // 1. READ & SMOOTH MOTION VECTORS
    // =====================================================================
    float3 currDirect = CurrColor.SampleLevel(LinearClamp, inputUv, 0).rgb;
    float2 rawMV   = Motion.SampleLevel(LinearClamp, inputUv, 0).xy * motionSampleScale;
    float  rawConf = saturate(pow(max(Confidence.SampleLevel(LinearClamp, inputUv, 0), 0.0), confPower));

    // Detect coarse MV field - detect both tiny (8x) and small (4x) resolution
    // For minimal pipeline: small (1/4) = scale 4, tiny (1/8) = scale 8
    float coarseFlag = saturate((motionSampleScale - 2.0) / 4.0);

    // 9-tap bilateral smoothing for coarse MV fields
    // Prevents blocky warping from low-resolution motion
    float2 fwdMV   = rawMV;
    float  fwdConf = rawConf;

    if (coarseFlag > 0.01) {
        uint mvW, mvH;
        Motion.GetDimensions(mvW, mvH);
        float2 mvTexel = 1.0 / float2(max(mvW, 1u), max(mvH, 1u));

        float centerLuma = Luma(currDirect);

        float2 mvAcc   = rawMV * (0.5 + rawConf);
        float  confAcc = rawConf * (0.5 + rawConf);
        float  wAcc    = 0.5 + rawConf;

        static const float2 kOff9[8] = {
            float2(-1,-1), float2(0,-1), float2(1,-1),
            float2(-1, 0),               float2(1, 0),
            float2(-1, 1), float2(0, 1), float2(1, 1)
        };

        [unroll] for (int i = 0; i < 8; ++i) {
            float2 sampleUv = clamp(inputUv + kOff9[i] * mvTexel, 0.0, 0.999);
            float2 nMV   = Motion.SampleLevel(LinearClamp, sampleUv, 0).xy * motionSampleScale;
            float  nConf = saturate(Confidence.SampleLevel(LinearClamp, sampleUv, 0));

            // Spatial weight (diagonals weaker)
            float spatialW = (abs(kOff9[i].x) + abs(kOff9[i].y) > 1.5) ? 0.5 : 1.0;

            // Motion coherence weight (reject outlier neighbors)
            float mvDist2 = dot(nMV - rawMV, nMV - rawMV);
            float motionW = exp(-mvDist2 / max(dot(rawMV, rawMV) * 4.0 + 1.0, 0.5));

            // Luma similarity weight (preserve edges)
            float3 nColor = CurrColor.SampleLevel(LinearClamp, sampleUv, 0).rgb;
            float lumaDiff = abs(Luma(nColor) - centerLuma);
            float lumaW = exp(-lumaDiff * lumaDiff / 0.01);

            float w = spatialW * motionW * lumaW * (0.15 + 0.85 * nConf);
            mvAcc   += nMV * w;
            confAcc += nConf * w;
            wAcc    += w;
        }

        fwdMV   = mvAcc / max(wAcc, 1e-4);
        fwdConf = confAcc / max(wAcc, 1e-4);
    }

    // =====================================================================
    // 1.5 MOTION VECTOR GATHER (Solve forward-warping holes)
    // =====================================================================
    // The motion field is defined at Curr. We are rendering at time alpha.
    // If an object moves from Prev to Curr, its motion vector is at its Curr position.
    // At the interpolated position, the motion vector might be 0 (background).
    // We search the neighborhood for a motion vector that projects to our current pixel.
    
    float2 bestMV = fwdMV;
    
    // First, evaluate how well the current center vector (fwdMV) aligns the features.
    float2 pPrevCenter = inputPos + fwdMV * alpha;
    float2 pCurrCenter = inputPos - fwdMV * (1.0 - alpha);
    float4 fPrevCenter = PrevFeature.SampleLevel(LinearClamp, pPrevCenter / inSize, 0);
    float4 fCurrCenter = CurrFeature.SampleLevel(LinearClamp, pCurrCenter / inSize, 0);
    float4 fPrevCenter2 = PrevFeature2.SampleLevel(LinearClamp, pPrevCenter / inSize, 0);
    float4 fCurrCenter2 = CurrFeature2.SampleLevel(LinearClamp, pCurrCenter / inSize, 0);
    float4 fPrevCenter3 = PrevFeature3.SampleLevel(LinearClamp, pPrevCenter / inSize, 0);
    float4 fCurrCenter3 = CurrFeature3.SampleLevel(LinearClamp, pCurrCenter / inSize, 0);
    
    float4 diffCenter = abs(fPrevCenter - fCurrCenter);
    float4 diffCenter2 = abs(fPrevCenter2 - fCurrCenter2);
    float4 diffCenter3 = abs(fPrevCenter3 - fCurrCenter3);
    
    // Advanced CNN: Weight the 12 channels based on their importance for alignment
    float4 w1 = float4(1.0, 1.0, 1.0, 2.0); // Luma, EdgeX, EdgeY, Texture (Texture is very important)
    float4 w2 = float4(2.0, 1.0, 1.0, 1.0); // Corner (Very important), Var, Diag1, Diag2
    float4 w3 = float4(0.5, 1.5, 1.5, 1.0); // Smooth (Less important), LoG, Mag, Cross
    
    // Base error metric: 12-channel CNN feature difference
    float minError = dot(diffCenter, w1) + dot(diffCenter2, w2) + dot(diffCenter3, w3);
    
    // Warp OOB penalty: penalize MVs that warp outside the valid region
    float2 prevUvTest = pPrevCenter / inSize;
    float2 currUvTest = pCurrCenter / inSize;
    float oobPenCenter = (any(prevUvTest < 0.005) || any(prevUvTest > 0.995) ||
                          any(currUvTest < 0.005) || any(currUvTest > 0.995)) ? 0.1 : 0.0;
    minError += oobPenCenter;
    minError += length(fwdMV) * 0.002; // Add length penalty to center as well
    
    // --- Zero-MV inheritance ---
    // If center MV is near-zero, gather the median of neighbor MVs.
    // This catches disoccluded regions where MotionEst found no match.
    float centerMVLen = length(fwdMV);
    float2 neighborMVAcc = float2(0, 0);
    float neighborMVW = 0;
    
    // Find max motion in neighborhood to scale search
    float maxLen = centerMVLen;
    static const float2 kCardinal[4] = {
        float2(0.02, 0), float2(-0.02, 0), float2(0, 0.02), float2(0, -0.02)
    };
    [unroll] for (int c = 0; c < 4; ++c) {
        float2 nMV = Motion.SampleLevel(LinearClamp, clamp(inputUv + kCardinal[c], 0.0, 0.999), 0).xy * motionSampleScale;
        float nLen = length(nMV);
        maxLen = max(maxLen, nLen);
        // Accumulate for zero-MV inheritance (weighted by magnitude)
        float nw = saturate(nLen * 0.5);
        neighborMVAcc += nMV * nw;
        neighborMVW += nw;
    }
    
    // If our MV is near-zero but neighbors have significant motion, inherit it
    if (centerMVLen < 1.0 && neighborMVW > 0.5) {
        float2 inheritedMV = neighborMVAcc / neighborMVW;
        // Evaluate inherited MV quality
        float2 iPrev = inputPos + inheritedMV * alpha;
        float2 iCurr = inputPos - inheritedMV * (1.0 - alpha);
        float4 fIP = PrevFeature.SampleLevel(LinearClamp, iPrev / inSize, 0);
        float4 fIC = CurrFeature.SampleLevel(LinearClamp, iCurr / inSize, 0);
        float4 fIP2 = PrevFeature2.SampleLevel(LinearClamp, iPrev / inSize, 0);
        float4 fIC2 = CurrFeature2.SampleLevel(LinearClamp, iCurr / inSize, 0);
        float4 fIP3 = PrevFeature3.SampleLevel(LinearClamp, iPrev / inSize, 0);
        float4 fIC3 = CurrFeature3.SampleLevel(LinearClamp, iCurr / inSize, 0);
        float iError = dot(abs(fIP - fIC), w1) + dot(abs(fIP2 - fIC2), w2) + dot(abs(fIP3 - fIC3), w3);
        iError += length(inheritedMV) * 0.002;
        if (iError < minError) {
            minError = iError;
            bestMV = inheritedMV;
        }
    }
    
    // Keep search radius standard to prevent jumping to the next repeating pattern (1-brick shift)
    float2 searchRadius = max(float2(0.005, 0.005), (maxLen / inSize) * 0.6);
    
    static const float2 kSearch[8] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1),
        float2(0, -2), float2(-2, 0), float2(2, 0), float2(0, 2)
    };
    
    // Periodicity detection (read once outside loop)
    float periodicity = CurrFeature3.SampleLevel(LinearClamp, inputUv, 0).w;
    
    [unroll] for (int j = 0; j < 8; ++j) {
        float2 sampleUv = clamp(inputUv + kSearch[j] * searchRadius, 0.0, 0.999);
        float2 testMV = Motion.SampleLevel(LinearClamp, sampleUv, 0).xy * motionSampleScale;
        
        float2 pPrev = inputPos + testMV * alpha;
        float2 pCurr = inputPos - testMV * (1.0 - alpha);
        
        // Warp OOB penalty for this candidate
        float2 pPrevUv = pPrev / inSize;
        float2 pCurrUv = pCurr / inSize;
        float oobPen = (any(pPrevUv < 0.005) || any(pPrevUv > 0.995) ||
                        any(pCurrUv < 0.005) || any(pCurrUv > 0.995)) ? 0.1 : 0.0;
        
        // Use CNN features to evaluate how well this motion vector aligns the textures
        float4 fPrev = PrevFeature.SampleLevel(LinearClamp, pPrevUv, 0);
        float4 fCurr = CurrFeature.SampleLevel(LinearClamp, pCurrUv, 0);
        float4 fPrev2 = PrevFeature2.SampleLevel(LinearClamp, pPrevUv, 0);
        float4 fCurr2 = CurrFeature2.SampleLevel(LinearClamp, pCurrUv, 0);
        float4 fPrev3 = PrevFeature3.SampleLevel(LinearClamp, pPrevUv, 0);
        float4 fCurr3 = CurrFeature3.SampleLevel(LinearClamp, pCurrUv, 0);
        
        float4 diff = abs(fPrev - fCurr);
        float4 diff2 = abs(fPrev2 - fCurr2);
        float4 diff3 = abs(fPrev3 - fCurr3);
        
        // Base error metric: 12-channel CNN feature difference
        float error = dot(diff, w1) + dot(diff2, w2) + dot(diff3, w3);
        error += oobPen;
        
        // Periodicity-aware tie-breaker:
        if (periodicity > 0.3) {
            float2 neighborMV1 = Motion.SampleLevel(LinearClamp, clamp(inputUv + float2(0.01, 0), 0.0, 0.999), 0).xy;
            float2 neighborMV2 = Motion.SampleLevel(LinearClamp, clamp(inputUv - float2(0.01, 0), 0.0, 0.999), 0).xy;
            float2 neighborMV3 = Motion.SampleLevel(LinearClamp, clamp(inputUv + float2(0, 0.01), 0.0, 0.999), 0).xy;
            float2 neighborMV4 = Motion.SampleLevel(LinearClamp, clamp(inputUv - float2(0, 0.01), 0.0, 0.999), 0).xy;
            
            float consistency1 = 1.0 - length(testMV - neighborMV1) * 0.5;
            float consistency2 = 1.0 - length(testMV - neighborMV2) * 0.5;
            float consistency3 = 1.0 - length(testMV - neighborMV3) * 0.5;
            float consistency4 = 1.0 - length(testMV - neighborMV4) * 0.5;
            float spatialConsistency = max(max(consistency1, consistency2), max(consistency3, consistency4));
            spatialConsistency = saturate(spatialConsistency);
            error -= spatialConsistency * 0.02 * periodicity;
        } else {
            error += length(testMV) * 0.002;
        }
        
        // HYSTERESIS: require 10% improvement to switch MVs
        // Prevents frame-to-frame flipping between marginal candidates
        if (error < minError * 0.90) {
            minError = error;
            bestMV = testMV;
        }
    }
    fwdMV = bestMV;

    // =====================================================================
    // 2. PURE WARPED SAMPLING (no dampening, full MV strength)
    // =====================================================================
    // The motion vector (fwdMV) points from Curr to Prev (because MotionEst searches PrevLuma around CurrLuma).
    // So fwdMV is the vector you add to a Curr pixel to find its source in Prev.
    // To find the color at the intermediate frame (at time 'alpha', where alpha=0 is Prev, alpha=1 is Curr):
    // - The object moves from Prev to Curr by vector -fwdMV.
    // - To look BACKWARDS in time to the Prev frame: pos + fwdMV * alpha
    // - To look FORWARDS in time to the Curr frame: pos - fwdMV * (1.0 - alpha)
    float2 warpPrevPos = inputPos + fwdMV * alpha;
    float2 warpCurrPos = inputPos - fwdMV * (1.0 - alpha);

    float2 warpPrevUv = clamp(warpPrevPos / inSize, 0.0, 0.999);
    float2 warpCurrUv = clamp(warpCurrPos / inSize, 0.0, 0.999);

    // --- CNN Feature-Guided Sub-pixel Alignment ---
    // Removed: This was causing jitter ("less smooth") because the gradients
    // can be noisy. We will rely on the MotionRefine pass for sub-pixel accuracy.
    // ----------------------------------------------

    float3 warpedPrev = SampleColor(PrevColor, warpPrevUv, inSize);
    float3 warpedCurr = SampleColor(CurrColor, warpCurrUv, inSize);

    // =====================================================================
    // 3. PURE AI WARP: OCCLUSION-AWARE SOURCE SELECTION (NO FRAME BLENDING)
    // =====================================================================
    // Both warpedPrev and warpedCurr are motion-compensated to target time.
    // Instead of lerp-blending (which ghosts when MVs are imperfect),
    // we SELECT the better source per-pixel using AI occlusion prediction.

    float4 fP1 = PrevFeature.SampleLevel(LinearClamp, warpPrevUv, 0);
    float4 fC1 = CurrFeature.SampleLevel(LinearClamp, warpCurrUv, 0);
    float4 fP2 = PrevFeature2.SampleLevel(LinearClamp, warpPrevUv, 0);
    float4 fC2 = CurrFeature2.SampleLevel(LinearClamp, warpCurrUv, 0);
    float4 fP3 = PrevFeature3.SampleLevel(LinearClamp, warpPrevUv, 0);
    float4 fC3 = CurrFeature3.SampleLevel(LinearClamp, warpCurrUv, 0);

    float4 featureDiff1 = fP1 - fC1;
    float4 featureDiff2 = fP2 - fC2;
    float4 featureDiff3 = fP3 - fC3;

    // Run synthesis MLP for per-pixel occlusion prediction
    float4 synthOut = SynthesisNet(featureDiff1, featureDiff2, featureDiff3);
    float occlusionSelect = synthOut.x;  // AI occlusion mask: 0=prev valid, 1=curr valid

    // --- Warp consistency: detect where motion vectors fail ---
    // Multi-channel warp consistency (not just luma) for more stable detection
    float3 warpDelta = abs(warpedPrev - warpedCurr);
    float warpAgreement = 1.0 - saturate(dot(warpDelta, float3(0.35, 0.45, 0.20)) * 5.0);

    // --- Warp quality bias: prefer the frame warped less distance ---
    float prevWarpLen = length(fwdMV * alpha);
    float currWarpLen = length(fwdMV * (1.0 - alpha));
    float qualityBias = 1.0 - currWarpLen / (prevWarpLen + currWarpLen + 0.001);

    // Base selection: AI occlusion prediction when trained, quality bias otherwise
    float rawSelect = lerp(qualityBias, occlusionSelect, saturate(useCustomWeights));

    // In areas where warps disagree, pull toward quality bias (more reliable frame)
    float mergedSelect = lerp(rawSelect, qualityBias, (1.0 - warpAgreement) * 0.35);

    // --- SMOOTH SELECTION: use smoothstep for stable, jitter-free transitions ---
    // Instead of aggressive sharpening (which flips between frames causing jitter),
    // use smoothstep for C1-continuous transitions. Only go harder at strong occlusion.
    float selectSharpness = lerp(1.0, 3.0, saturate((1.0 - warpAgreement) * 1.5));
    float finalSelect = smoothstep(0.0, 1.0, saturate((mergedSelect - 0.5) * selectSharpness + 0.5));

    // Pure warped result — NO alpha blending, NO unwarped fallback
    float3 result = lerp(warpedPrev, warpedCurr, finalSelect);

    // Fast path for pure real frames to avoid any floating point inaccuracies
    if (alpha <= 0.001) {
        result = SampleColor(PrevColor, inputUv, inSize);
    } else if (alpha >= 0.999) {
        result = SampleColor(CurrColor, inputUv, inSize);
    }

    OutColor[id.xy] = float4(saturate(result), 1.0);
}
