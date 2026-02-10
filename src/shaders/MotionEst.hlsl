// ============================================================================
// MOTION ESTIMATION - Robust Block Matching
// Tuned to reduce ambiguous vectors that create visible warp artifacts.
// ============================================================================

Texture2D<float> CurrLuma : register(t0);
Texture2D<float> PrevLuma : register(t1);
Texture2D<float2> MotionPred : register(t2);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfidenceOut : register(u1);

SamplerState LinearClamp : register(s0);

cbuffer MotionCB : register(b0) {
    int radius;
    int usePrediction;
    float predictionScale;
    float pad;
};

#define BLOCK_SIZE 16
#define BLOCK_R 8
#define BLOCK_EXTENT (BLOCK_SIZE + BLOCK_R * 2)

groupshared float gs_Curr[BLOCK_EXTENT][BLOCK_EXTENT];

float BlockWeight(int2 offset) {
    float dist = max(abs(offset.x), abs(offset.y));
    float w = 1.0 - dist / (float(BLOCK_R) + 1.0);
    return 0.25 + 0.75 * saturate(w);
}

float RobustDiff(float a, float b) {
    return min(abs(a - b), 0.35);
}

float EvalSadInt(int2 pos, int2 localPos, int2 mv, uint w, uint h) {
    float sad = 0.0;
    int2 maxPos = int2(int(w) - 1, int(h) - 1);

    [loop]
    for (int by = -BLOCK_R; by <= BLOCK_R; ++by) {
        [loop]
        for (int bx = -BLOCK_R; bx <= BLOCK_R; ++bx) {
            int2 offset = int2(bx, by);
            int2 sp = localPos + offset;
            float cVal = gs_Curr[sp.y][sp.x];
            int2 pPos = clamp(pos + offset + mv, int2(0, 0), maxPos);
            float pVal = PrevLuma.Load(int3(pPos, 0));
            sad += RobustDiff(cVal, pVal) * BlockWeight(offset);
        }
    }

    return sad;
}

float EvalSadFrac(int2 pos, int2 localPos, float2 mv, float2 invSize) {
    float sad = 0.0;

    [loop]
    for (int by = -BLOCK_R; by <= BLOCK_R; ++by) {
        [loop]
        for (int bx = -BLOCK_R; bx <= BLOCK_R; ++bx) {
            int2 offset = int2(bx, by);
            int2 sp = localPos + offset;
            float cVal = gs_Curr[sp.y][sp.x];
            float2 pPos = float2(pos + offset) + 0.5 + mv;
            float2 pUv = clamp(pPos * invSize, 0.0, 0.999);
            float pVal = PrevLuma.SampleLevel(LinearClamp, pUv, 0);
            sad += RobustDiff(cVal, pVal) * BlockWeight(offset);
        }
    }

    return sad;
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

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint w, h;
    CurrLuma.GetDimensions(w, h);

    int2 groupBase = int2(gid.xy) * BLOCK_SIZE - BLOCK_R;
    uint tid = gtid.y * BLOCK_SIZE + gtid.x;

    // Load current frame tile + apron into shared memory.
    [loop]
    for (int i = 0; i < 4; ++i) {
        uint pIdx = tid + i * 256;
        if (pIdx < 1024) {
            int ly = int(pIdx / BLOCK_EXTENT);
            int lx = int(pIdx % BLOCK_EXTENT);
            int2 loadPos = clamp(groupBase + int2(lx, ly), int2(0, 0), int2(int(w) - 1, int(h) - 1));
            gs_Curr[ly][lx] = CurrLuma.Load(int3(loadPos, 0));
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (id.x >= w || id.y >= h) {
        return;
    }

    int2 pos = int2(id.xy);
    int2 localPos = int2(gtid.xy) + BLOCK_R;
    float2 invSize = 1.0 / float2(w, h);
    float2 uv = (float2(pos) + 0.5) * invSize;

    float currCenter = gs_Curr[localPos.y][localPos.x];
    float prevCenter = PrevLuma.Load(int3(pos, 0));
    float frameDiff = abs(currCenter - prevCenter);

    int lxL = max(localPos.x - 1, 0);
    int lxR = min(localPos.x + 1, BLOCK_EXTENT - 1);
    int lyU = max(localPos.y - 1, 0);
    int lyD = min(localPos.y + 1, BLOCK_EXTENT - 1);
    float gx = abs(gs_Curr[localPos.y][lxR] - gs_Curr[localPos.y][lxL]);
    float gy = abs(gs_Curr[lyD][localPos.x] - gs_Curr[lyU][localPos.x]);
    float textureStrength = saturate((gx + gy) * 6.0);

    int maxR = max(1, int(min(w, h) / 4));
    int baseR = clamp(radius, 1, maxR);
    float motionHint = max(smoothstep(0.01, 0.2, frameDiff), textureStrength * 0.75);
    int searchR = clamp(int(round(lerp(float(baseR), float(baseR) * 1.5, motionHint))), 1, maxR);

    float bestSad = 1e9;
    float secondSad = 1e9;
    float2 bestMV = float2(0.0, 0.0);

    float motionPenaltyBase = lerp(0.16, 0.04, textureStrength);

    // Zero motion candidate.
    float zeroSad = EvalSadInt(pos, localPos, int2(0, 0), w, h);
    float zeroBias = lerp(0.88, 0.98, textureStrength);
    UpdateBest(zeroSad * zeroBias, float2(0.0, 0.0), bestSad, bestMV, secondSad);

    // Predicted candidate.
    if (usePrediction != 0) {
        float2 pred = MotionPred.SampleLevel(LinearClamp, uv, 0).xy * predictionScale;
        pred = clamp(pred, -float2(searchR, searchR), float2(searchR, searchR));
        if (dot(pred, pred) > 0.05) {
            int2 predMV = int2(round(pred));
            float predSad = EvalSadInt(pos, localPos, predMV, w, h);
            predSad += length(float2(predMV)) * motionPenaltyBase;
            UpdateBest(predSad, float2(predMV), bestSad, bestMV, secondSad);
        }
    }

    static const int2 kDiamond[4] = {
        int2(0, -1),
        int2(1, 0),
        int2(0, 1),
        int2(-1, 0)
    };

    // Coarse-to-fine diamond search.
    float2 searchCenter = bestMV;
    uint step = max(1u, (uint(searchR) + 1u) >> 1u);
    while (step >= 1u) {
        bool foundBetter = false;
        float stepBestSad = bestSad;
        float2 stepBestMV = searchCenter;
        int2 centerMV = int2(round(searchCenter));

        [unroll]
        for (int d = 0; d < 4; ++d) {
            int2 testMV = centerMV + kDiamond[d] * int(step);
            testMV = clamp(testMV, -int2(searchR, searchR), int2(searchR, searchR));

            float sad = EvalSadInt(pos, localPos, testMV, w, h);
            sad += length(float2(testMV)) * motionPenaltyBase;
            UpdateBest(sad, float2(testMV), bestSad, bestMV, secondSad);

            if (sad < stepBestSad) {
                stepBestSad = sad;
                stepBestMV = float2(testMV);
                foundBetter = true;
            }
        }

        if (foundBetter) {
            searchCenter = stepBestMV;
        } else {
            if (step == 1u) {
                break;
            }
            step >>= 1u;
        }
    }

    // Integer local refinement.
    int2 bestIntMV = int2(round(bestMV));
    [loop]
    for (int dyI = -1; dyI <= 1; ++dyI) {
        [loop]
        for (int dxI = -1; dxI <= 1; ++dxI) {
            int2 testMV = bestIntMV + int2(dxI, dyI);
            testMV = clamp(testMV, -int2(searchR, searchR), int2(searchR, searchR));
            float sad = EvalSadInt(pos, localPos, testMV, w, h);
            sad += length(float2(testMV)) * motionPenaltyBase;
            UpdateBest(sad, float2(testMV), bestSad, bestMV, secondSad);
        }
    }

    // Half-pixel refinement.
    float2 halfCenter = bestMV;
    [loop]
    for (int dyH = -1; dyH <= 1; ++dyH) {
        [loop]
        for (int dxH = -1; dxH <= 1; ++dxH) {
            if (dxH == 0 && dyH == 0) {
                continue;
            }
            float2 testMV = halfCenter + float2(dxH, dyH) * 0.5;
            testMV = clamp(testMV, -float2(searchR, searchR), float2(searchR, searchR));
            float sad = EvalSadFrac(pos, localPos, testMV, invSize);
            sad += length(testMV) * (motionPenaltyBase * 0.75);
            UpdateBest(sad, testMV, bestSad, bestMV, secondSad);
        }
    }

    // Quarter-pixel refinement.
    float2 quarterCenter = bestMV;
    [loop]
    for (int dyQ = -1; dyQ <= 1; ++dyQ) {
        [loop]
        for (int dxQ = -1; dxQ <= 1; ++dxQ) {
            if (dxQ == 0 && dyQ == 0) {
                continue;
            }
            float2 testMV = quarterCenter + float2(dxQ, dyQ) * 0.25;
            testMV = clamp(testMV, -float2(searchR, searchR), float2(searchR, searchR));
            float sad = EvalSadFrac(pos, localPos, testMV, invSize);
            sad += length(testMV) * (motionPenaltyBase * 0.6);
            UpdateBest(sad, testMV, bestSad, bestMV, secondSad);
        }
    }

    float uniqueness = saturate((secondSad - bestSad) / max(secondSad, 1e-4));
    float staticRegion = 1.0 - smoothstep(0.02, 0.14, frameDiff);
    float ambiguity = 1.0 - uniqueness;
    float damping = ambiguity * (1.0 - textureStrength) * staticRegion;
    bestMV *= (1.0 - 0.70 * damping);

    float photoSad = EvalSadFrac(pos, localPos, bestMV, invSize);
    float sampleCount = float((2 * BLOCK_R + 1) * (2 * BLOCK_R + 1));
    float avgDiff = photoSad / sampleCount;
    float matchConf = exp(-avgDiff * 9.0);
    float confidence = matchConf;
    confidence *= (0.35 + 0.65 * uniqueness);
    confidence *= lerp(0.55, 1.0, textureStrength);
    if (frameDiff < 0.02 && length(bestMV) < 0.5) {
        confidence = max(confidence, 0.90);
    }
    confidence = clamp(confidence, 0.05, 0.98);

    MotionOut[id.xy] = bestMV;
    ConfidenceOut[id.xy] = confidence;
}
