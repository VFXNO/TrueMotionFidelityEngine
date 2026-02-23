// ============================================================================
// FRAME INTERPOLATION v2.2 - Pure Motion-Compensated Warping
//
// Architecture:
//   1. Read and smooth motion vectors (9-tap bilateral on coarse fields)
//   2. Forward warp from Prev, backward warp from Curr
//   3. Alpha-weighted blend between warped prev and warped curr
//   4. No dissolve, no blending fallback - purely warped output
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

    // Detect coarse MV field (minimal pipeline uses tiny resolution)
    float coarseFlag = saturate((motionSampleScale - 2.5) / 3.5);

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
    minError += length(fwdMV) * 0.002; // Add length penalty to center as well
    
    // Find max motion in neighborhood to scale search
    float maxLen = length(fwdMV);
    maxLen = max(maxLen, length(Motion.SampleLevel(LinearClamp, clamp(inputUv + float2(0.02, 0), 0.0, 0.999), 0).xy * motionSampleScale));
    maxLen = max(maxLen, length(Motion.SampleLevel(LinearClamp, clamp(inputUv + float2(-0.02, 0), 0.0, 0.999), 0).xy * motionSampleScale));
    maxLen = max(maxLen, length(Motion.SampleLevel(LinearClamp, clamp(inputUv + float2(0, 0.02), 0.0, 0.999), 0).xy * motionSampleScale));
    maxLen = max(maxLen, length(Motion.SampleLevel(LinearClamp, clamp(inputUv + float2(0, -0.02), 0.0, 0.999), 0).xy * motionSampleScale));
    
    // Keep search radius standard to prevent jumping to the next repeating pattern (1-brick shift)
    float2 searchRadius = max(float2(0.005, 0.005), (maxLen / inSize) * 0.6);
    
    static const float2 kSearch[8] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1),
        float2(0, -2), float2(-2, 0), float2(2, 0), float2(0, 2)
    };
    
    [unroll] for (int j = 0; j < 8; ++j) {
        float2 sampleUv = clamp(inputUv + kSearch[j] * searchRadius, 0.0, 0.999);
        float2 testMV = Motion.SampleLevel(LinearClamp, sampleUv, 0).xy * motionSampleScale;
        
        float2 pPrev = inputPos + testMV * alpha;
        float2 pCurr = inputPos - testMV * (1.0 - alpha);
        
        // Use CNN features to evaluate how well this motion vector aligns the textures
        float4 fPrev = PrevFeature.SampleLevel(LinearClamp, pPrev / inSize, 0);
        float4 fCurr = CurrFeature.SampleLevel(LinearClamp, pCurr / inSize, 0);
        float4 fPrev2 = PrevFeature2.SampleLevel(LinearClamp, pPrev / inSize, 0);
        float4 fCurr2 = CurrFeature2.SampleLevel(LinearClamp, pCurr / inSize, 0);
        float4 fPrev3 = PrevFeature3.SampleLevel(LinearClamp, pPrev / inSize, 0);
        float4 fCurr3 = CurrFeature3.SampleLevel(LinearClamp, pCurr / inSize, 0);
        
        float4 diff = abs(fPrev - fCurr);
        float4 diff2 = abs(fPrev2 - fCurr2);
        float4 diff3 = abs(fPrev3 - fCurr3);
        
        // Base error metric: 12-channel CNN feature difference
        float error = dot(diff, w1) + dot(diff2, w2) + dot(diff3, w3);
        
        // Tie-breaker: prefer smaller motion vectors (background) to avoid 
        // jumping to a foreground vector on repeating texture patterns.
        error += length(testMV) * 0.002;
        
        if (error < minError) {
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
    // 3. CNN FEATURE-AWARE BLENDING & DETAIL INJECTION
    // =====================================================================
    // Sample the 12-channel CNN features at the final warped locations
    float4 fP1 = PrevFeature.SampleLevel(LinearClamp, warpPrevUv, 0);
    float4 fC1 = CurrFeature.SampleLevel(LinearClamp, warpCurrUv, 0);
    float4 fP2 = PrevFeature2.SampleLevel(LinearClamp, warpPrevUv, 0);
    float4 fC2 = CurrFeature2.SampleLevel(LinearClamp, warpCurrUv, 0);
    float4 fP3 = PrevFeature3.SampleLevel(LinearClamp, warpPrevUv, 0);
    float4 fC3 = CurrFeature3.SampleLevel(LinearClamp, warpCurrUv, 0);

    // Calculate the "structural energy" of the warped patches
    // Feature 11 is Edge Magnitude (fP3.z and fC3.z)
    // Feature 5 is Corner Response (fP2.x and fC2.x)
    // Feature 6 is Local Variance (fP2.y and fC2.y)
    float energyPrev = fP3.z + abs(fP2.x) + fP2.y;
    float energyCurr = fC3.z + abs(fC2.x) + fC2.y;

    // If one frame has significantly more structural energy (e.g. it's not occluded/blurry),
    // we shift the blend weight slightly towards it to preserve details.
    float energyDiff = energyPrev - energyCurr;
    float energyWeight = saturate(0.5 + energyDiff * 0.5); // 0.5 is neutral
    
    // Combine the temporal alpha with the structural energy weight
    // We don't want to completely override alpha, just bias it by up to 30%
    // Fade out the energy bias near alpha=0 and alpha=1 to ensure pure real frames
    float biasStrength = 0.3 * (1.0 - abs(alpha * 2.0 - 1.0)); // 0 at alpha=0/1, 0.3 at alpha=0.5
    float blendWeight = lerp(alpha, energyWeight, biasStrength);

    // We use the feature-aware blend to ensure we never "cancel" the warping,
    // but intelligently preserve sharp edges and corners during the warp.
    float3 result = lerp(warpedPrev, warpedCurr, blendWeight);

    // Feature-Guided Detail Injection:
    // Extract high-frequency details (Texture Pattern + LoG) from the CNN
    float detailPrev = fP1.w + fP3.y;
    float detailCurr = fC1.w + fC3.y;
    
    // Blend the details using the same weight
    float blendedDetail = lerp(detailPrev, detailCurr, blendWeight);
    
    // Add the details back to the luma of the result to recover sharpness lost during bicubic warping
    // Fade out sharpening near alpha=0 and alpha=1 to avoid altering real frames
    result += blendedDetail * 0.05 * (1.0 - abs(alpha * 2.0 - 1.0));

    // Fast path for pure real frames to avoid any floating point inaccuracies
    if (alpha <= 0.001) {
        result = SampleColor(PrevColor, inputUv, inSize);
    } else if (alpha >= 0.999) {
        result = SampleColor(CurrColor, inputUv, inSize);
    }

    OutColor[id.xy] = float4(saturate(result), 1.0);
}
