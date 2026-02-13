// ============================================================================
// MOTION REFINEMENT - Robust local refinement with consistency checks
// Tuned for lower cost while preserving visual quality.
// ============================================================================

Texture2D<float> CurrLuma : register(t0);
Texture2D<float> PrevLuma : register(t1);
Texture2D<float2> CoarseMotion : register(t2);
Texture2D<float> CoarseConf : register(t3);
Texture2D<float2> BackwardMotion : register(t4);
Texture2D<float> BackwardConf : register(t5);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfidenceOut : register(u1);

SamplerState LinearClamp : register(s0);

cbuffer RefineCB : register(b0) {
    int radius;
    float motionScale;
    int useBackward;
    float backwardScale;
};

#define REFINE_R_INT 2
#define REFINE_R_FRAC 2

static const float kKernel5x5[25] = {
    0.30, 0.30, 0.30, 0.30, 0.30,
    0.30, 0.55, 0.55, 0.55, 0.30,
    0.30, 0.55, 1.00, 0.55, 0.30,
    0.30, 0.55, 0.55, 0.55, 0.30,
    0.30, 0.30, 0.30, 0.30, 0.30
};

float KernelWeight5x5(int bx, int by) {
    return kKernel5x5[(by + 2) * 5 + (bx + 2)];
}

float RobustDiff(float a, float b) {
    return min(abs(a - b), 0.30);
}

float MotionPenalty(float2 mv) {
    float2 a = abs(mv);
    float hi = max(a.x, a.y);
    float lo = min(a.x, a.y);
    return hi + lo * 0.5;
}

float EvalSadInt(int2 pos, int2 mv, uint w, uint h, float cutoff) {
    float sad = 0.0;
    int2 maxPos = int2(int(w) - 1, int(h) - 1);

    [loop]
    for (int by = -REFINE_R_INT; by <= REFINE_R_INT; ++by) {
        [loop]
        for (int bx = -REFINE_R_INT; bx <= REFINE_R_INT; ++bx) {
            int2 cPos = clamp(pos + int2(bx, by), int2(0, 0), maxPos);
            int2 pPos = clamp(pos + int2(bx, by) + mv, int2(0, 0), maxPos);
            float cVal = CurrLuma.Load(int3(cPos, 0));
            float pVal = PrevLuma.Load(int3(pPos, 0));
            sad += RobustDiff(cVal, pVal) * KernelWeight5x5(bx, by);
            if (sad > cutoff) {
                return sad;
            }
        }
    }

    return sad;
}

float EvalSadFrac(int2 pos, float2 mv, uint w, uint h, float2 invSize, float cutoff) {
    float sad = 0.0;
    int2 maxPos = int2(int(w) - 1, int(h) - 1);

    [loop]
    for (int by = -REFINE_R_FRAC; by <= REFINE_R_FRAC; ++by) {
        [loop]
        for (int bx = -REFINE_R_FRAC; bx <= REFINE_R_FRAC; ++bx) {
            int2 cPos = clamp(pos + int2(bx, by), int2(0, 0), maxPos);
            float cVal = CurrLuma.Load(int3(cPos, 0));

            float2 pPos = float2(cPos) + 0.5 + mv;
            float2 pUv = clamp(pPos * invSize, 0.0, 0.999);
            float pVal = PrevLuma.SampleLevel(LinearClamp, pUv, 0);

            sad += RobustDiff(cVal, pVal) * KernelWeight5x5(bx, by);
            if (sad > cutoff) {
                return sad;
            }
        }
    }

    return sad;
}

float EvalBackwardPenalty(float2 mv, int2 pos, float2 invSize) {
    if (useBackward == 0) {
        return 0.0;
    }

    float2 matchPos = float2(pos) + 0.5 + mv;
    float2 matchUv = clamp(matchPos * invSize, 0.0, 0.999);
    float2 backMV = BackwardMotion.SampleLevel(LinearClamp, matchUv, 0).xy * backwardScale;
    float backConf = saturate(BackwardConf.SampleLevel(LinearClamp, matchUv, 0));
    float consistency = MotionPenalty(mv + backMV);
    return consistency * lerp(0.06, 0.20, backConf);
}

void UpdateBest(float cost, float2 mv, inout float bestCost, inout float2 bestMV, inout float secondCost) {
    if (cost < bestCost) {
        secondCost = bestCost;
        bestCost = cost;
        bestMV = mv;
    } else if (cost < secondCost) {
        secondCost = cost;
    }
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    CurrLuma.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) {
        return;
    }

    int2 pos = int2(id.xy);
    float2 invSize = 1.0 / float2(w, h);
    float2 uv = (float2(pos) + 0.5) * invSize;

    float2 coarseMV = CoarseMotion.SampleLevel(LinearClamp, uv, 0).xy * motionScale;
    float coarseConf = saturate(CoarseConf.SampleLevel(LinearClamp, uv, 0));

    int searchR = clamp(radius, 1, 4);
    if (coarseConf > 0.85) {
        searchR = min(searchR, 2);
    } else if (coarseConf > 0.70) {
        searchR = min(searchR, 3);
    }
    float regWeight = lerp(0.10, 0.03, coarseConf);

    float bestSad = 1e9;
    float secondSad = 1e9;
    float2 bestMV = coarseMV;

    // Fast-path: high-confidence near-static coarse vectors usually do not need expensive refinement.
    if (coarseConf > 0.94 && dot(coarseMV, coarseMV) < 0.04) {
        MotionOut[id.xy] = coarseMV;
        ConfidenceOut[id.xy] = max(coarseConf, 0.95);
        return;
    }

    float coarseSad = EvalSadFrac(pos, coarseMV, w, h, invSize, 1e30);
    coarseSad += EvalBackwardPenalty(coarseMV, pos, invSize);
    UpdateBest(coarseSad, coarseMV, bestSad, bestMV, secondSad);

    // Integer neighborhood around coarse estimate.
    int2 baseMV = int2(round(coarseMV));
    [loop]
    for (int dyI = -searchR; dyI <= searchR; ++dyI) {
        [loop]
        for (int dxI = -searchR; dxI <= searchR; ++dxI) {
            int2 testMV = baseMV + int2(dxI, dyI);
            float2 testMVf = float2(testMV);
            float sad = EvalSadInt(pos, testMV, w, h, secondSad);
            if (sad >= secondSad) {
                continue;
            }
            sad += MotionPenalty(testMVf - coarseMV) * regWeight;
            if (sad >= secondSad) {
                continue;
            }
            sad += EvalBackwardPenalty(testMVf, pos, invSize);
            UpdateBest(sad, testMVf, bestSad, bestMV, secondSad);
        }
    }

    // Half-pixel refinement around current best.
    float halfStartSad = bestSad;
    float2 halfCenter = bestMV;
    float2 mvLimit = float2(searchR + 1, searchR + 1);
    [loop]
    for (int dyH = -1; dyH <= 1; ++dyH) {
        [loop]
        for (int dxH = -1; dxH <= 1; ++dxH) {
            if (dxH == 0 && dyH == 0) {
                continue;
            }

            float2 testMV = halfCenter + float2(dxH, dyH) * 0.5;
            testMV = clamp(testMV, coarseMV - mvLimit, coarseMV + mvLimit);
            float sad = EvalSadFrac(pos, testMV, w, h, invSize, secondSad);
            if (sad >= secondSad) {
                continue;
            }
            sad += MotionPenalty(testMV - coarseMV) * (regWeight * 0.75);
            if (sad >= secondSad) {
                continue;
            }
            sad += EvalBackwardPenalty(testMV, pos, invSize);
            UpdateBest(sad, testMV, bestSad, bestMV, secondSad);
        }
    }

    // Quarter-pixel refinement can be skipped on stable converged pixels.
    float halfGain = halfStartSad - bestSad;
    bool runQuarter = (halfGain > (halfStartSad * 0.003)) || (coarseConf < 0.70) || (dot(bestMV - coarseMV, bestMV - coarseMV) > 0.04);
    if (runQuarter) {
        float2 quarterCenter = bestMV;
        [loop]
        for (int dyQ = -1; dyQ <= 1; ++dyQ) {
            [loop]
            for (int dxQ = -1; dxQ <= 1; ++dxQ) {
                if (dxQ == 0 && dyQ == 0) {
                    continue;
                }

                float2 testMV = quarterCenter + float2(dxQ, dyQ) * 0.25;
                testMV = clamp(testMV, coarseMV - mvLimit, coarseMV + mvLimit);
                float sad = EvalSadFrac(pos, testMV, w, h, invSize, secondSad);
                if (sad >= secondSad) {
                    continue;
                }
                sad += MotionPenalty(testMV - coarseMV) * (regWeight * 0.6);
                if (sad >= secondSad) {
                    continue;
                }
                sad += EvalBackwardPenalty(testMV, pos, invSize);
                UpdateBest(sad, testMV, bestSad, bestMV, secondSad);
            }
        }
    }

    float uniqueness = saturate((secondSad - bestSad) / max(secondSad, 1e-4));
    float ambiguity = 1.0 - uniqueness;
    float snapBack = ambiguity * (1.0 - coarseConf);
    bestMV = lerp(bestMV, coarseMV, snapBack * 0.6);

    float photoSad = EvalSadFrac(pos, bestMV, w, h, invSize, 1e30);
    float sampleCount = float((2 * REFINE_R_FRAC + 1) * (2 * REFINE_R_FRAC + 1));
    float avgDiff = photoSad / sampleCount;
    float matchConf = exp(-avgDiff * 8.0);
    float confidence = matchConf * (0.4 + 0.6 * uniqueness);
    confidence = lerp(confidence, coarseConf, 0.35);
    confidence = clamp(confidence, 0.05, 0.98);

    MotionOut[id.xy] = bestMV;
    ConfidenceOut[id.xy] = confidence;
}
