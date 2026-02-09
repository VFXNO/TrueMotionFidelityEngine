// ============================================================================
// MOTION ESTIMATION - High Quality SAD Block Matching
// Large search radius with sub-pixel refinement for smooth video interpolation
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
#define SEARCH_R 8

groupshared float gs_Curr[32][32];

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint w, h;
    CurrLuma.GetDimensions(w, h);
    
    int2 groupBase = gid.xy * BLOCK_SIZE - BLOCK_R;
    int2 loadPosBase = groupBase; // Base for loading
    uint tid = gtid.y * BLOCK_SIZE + gtid.x;
    
    // Load Current Block + Apron into Shared Memory (32x32 = 1024 pixels)
    // 256 threads * 4 iter = 1024 loads. Coverage: Perfect.
    for (int i = 0; i < 4; ++i) {
        uint pIdx = tid + i * 256;
        if (pIdx < 1024) {
            int ly = pIdx / 32;
            int lx = pIdx % 32;
            int2 loadPos = clamp(groupBase + int2(lx, ly), int2(0, 0), int2(w - 1, h - 1));
            gs_Curr[ly][lx] = CurrLuma.Load(int3(loadPos, 0));
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    if (id.x >= w || id.y >= h) return;
    
    int2 pos = int2(id.xy);
    int2 localPos = int2(gtid.xy) + BLOCK_R;
    
    float2 uv = (float2(pos) + 0.5) / float2(w, h);
    float currCenter = gs_Curr[localPos.y][localPos.x];
    // Load from Texture directly (handles arbitrary radius)
    float prevCenter = PrevLuma.Load(int3(pos, 0));
    float frameDiff = abs(currCenter - prevCenter);
    
    int maxR = int(min(w, h) / 4);
    int baseR = clamp(radius, SEARCH_R, maxR);
    
    float motionHint = smoothstep(0.0, 0.2, frameDiff);
    int searchR = int(lerp(float(baseR), float(baseR) * 1.5, motionHint));
    searchR = min(searchR, maxR);
    
    float bestSad = 1e9f;
    float2 bestMV = float2(0, 0);
    
    // 1. Check Zero Motion
    float sad = 0.0f;
    for (int by = -BLOCK_R; by <= BLOCK_R; ++by) {
        for (int bx = -BLOCK_R; bx <= BLOCK_R; ++bx) {
            int2 sp = localPos + int2(bx, by);
            float cVal = gs_Curr[sp.y][sp.x];
            
            int2 pPos = clamp(pos + int2(bx, by), int2(0,0), int2(w-1, h-1));
            float pVal = PrevLuma.Load(int3(pPos, 0));
            
            // Weighted SAD: Center pixels matter more to prevent background halos
            float dist = max(abs(bx), abs(by));
            float weight = 1.0 - (dist / (float(BLOCK_R) + 2.0)); 
            sad += abs(cVal - pVal) * weight;
        }
    }
    bestSad = sad * 0.95f; // Bias for zero motion
    
    // 2. Prediction
    if (usePrediction) {
        float2 pred = MotionPred.SampleLevel(LinearClamp, uv, 0).xy * predictionScale;
        float2 clampedPred = clamp(pred, -float2(searchR, searchR), float2(searchR, searchR));
        
        if (length(clampedPred) > 0.5f) {
            int2 mvi = int2(round(clampedPred));
            sad = 0.0f;
            for (int byP = -BLOCK_R; byP <= BLOCK_R; ++byP) {
                for (int bxP = -BLOCK_R; bxP <= BLOCK_R; ++bxP) {
                    int2 sp = localPos + int2(bxP, byP);
                    float cVal = gs_Curr[sp.y][sp.x];
                    
                    int2 pPos = clamp(pos + int2(bxP, byP) + mvi, int2(0,0), int2(w-1, h-1));
                    float pVal = PrevLuma.Load(int3(pPos, 0));
                    
                    float dist = max(abs(bxP), abs(byP));
                    float weight = 1.0 - (dist / (float(BLOCK_R) + 2.0)); 
                    sad += abs(cVal - pVal) * weight;
                }
            }
            if (sad < bestSad * 1.05f) {
                bestSad = sad;
                bestMV = clampedPred;
            }
        }
    }
    
    static const int2 kDiamond[4] = { int2(0, -1), int2(1, 0), int2(0, 1), int2(-1, 0) };
    
    float2 searchCenter = bestMV;
    uint step = max(2u, uint(searchR) >> 1u);
    
    // 3. Diamond Search
    while (step >= 1u) {
        float2 stepBestMV = searchCenter;
        float stepBestSad = bestSad;
        bool foundBetter = false;

        for (int d = 0; d < 4; ++d) {
            int2 testMV = int2(searchCenter) + kDiamond[d] * int(step);
            testMV = clamp(testMV, -int2(maxR, maxR), int2(maxR, maxR));
            
            sad = 0.0f;
            for (int byD = -BLOCK_R; byD <= BLOCK_R; ++byD) {
                for (int bxD = -BLOCK_R; bxD <= BLOCK_R; ++bxD) {
                    int2 sp = localPos + int2(bxD, byD);
                    float cVal = gs_Curr[sp.y][sp.x];
                    
                    int2 pPos = clamp(pos + int2(bxD, byD) + testMV, int2(0,0), int2(w-1, h-1));
                    float pVal = PrevLuma.Load(int3(pPos, 0));
                    
                    float dist = max(abs(bxD), abs(byD));
                    float weight = 1.0 - (dist / (float(BLOCK_R) + 2.0)); 
                    sad += abs(cVal - pVal) * weight;
                }
            }
            
            sad += length(float2(testMV)) * 0.05f;
            
            if (sad < stepBestSad) {
                stepBestSad = sad;
                stepBestMV = float2(testMV);
                foundBetter = true;
            }
        }
        
        if (foundBetter) {
            bestSad = stepBestSad;
            searchCenter = stepBestMV;
        } else {
             step >>= 1u;
        }
        bestMV = searchCenter;
    }
    
    // 4. Integer Refinement
    float2 bestIntMV = bestMV;
    for (int dyI = -1; dyI <= 1; ++dyI) {
        for (int dxI = -1; dxI <= 1; ++dxI) {
            if (dxI == 0 && dyI == 0) continue;
            
            int2 testMV = int2(round(bestMV)) + int2(dxI, dyI);
            testMV = clamp(testMV, -int2(maxR, maxR), int2(maxR, maxR));
            
            sad = 0.0f;
            for (int byI = -BLOCK_R; byI <= BLOCK_R; ++byI) {
                for (int bxI = -BLOCK_R; bxI <= BLOCK_R; ++bxI) {
                     int2 sp = localPos + int2(bxI, byI);
                     float cVal = gs_Curr[sp.y][sp.x];
                     
                     int2 pPos = clamp(pos + int2(bxI, byI) + testMV, int2(0,0), int2(w-1, h-1));
                     float pVal = PrevLuma.Load(int3(pPos, 0));
                     sad += abs(cVal - pVal);
                }
            }
            
            sad += length(float2(testMV)) * 0.05f;
            
            if (sad < bestSad) {
                bestSad = sad;
                bestIntMV = float2(testMV);
            }
        }
    }
    
    // 5. Half-Pixel Refinement (Linear Sample)
    float2 bestHalfMV = bestIntMV;
    float bestHalfSad = bestSad;
    
    for (int dyH = -1; dyH <= 1; ++dyH) {
        for (int dxH = -1; dxH <= 1; ++dxH) {
            if (dxH == 0 && dyH == 0) continue;
            
            float2 testMV = bestIntMV + float2(dxH, dyH) * 0.5f;
            
            sad = 0.0f;
            for (int byH = -BLOCK_R; byH <= BLOCK_R; ++byH) {
                for (int bxH = -BLOCK_R; bxH <= BLOCK_R; ++bxH) {
                    int2 sp = localPos + int2(bxH, byH);
                    float cVal = gs_Curr[sp.y][sp.x];
                    
                    float2 pPos = float2(pos) + float2(bxH, byH) + 0.5f + testMV;
                    float2 pUV = pPos / float2(w, h); // Use normalized coords for SampleLevel
                    
                    // Linear Filtered Sample
                    sad += abs(cVal - PrevLuma.SampleLevel(LinearClamp, pUV, 0));
                }
            }
            
            if (sad < bestHalfSad) {
                bestHalfSad = sad;
                bestHalfMV = testMV;
            }
        }
    }
    
    // 6. Quarter-Pixel Refinement
    float2 bestQuarterMV = bestHalfMV;
    float bestQuarterSad = bestHalfSad;
    
    for (int dyQ = -1; dyQ <= 1; ++dyQ) {
        for (int dxQ = -1; dxQ <= 1; ++dxQ) {
            if (dxQ == 0 && dyQ == 0) continue;
            
            float2 testMV = bestHalfMV + float2(dxQ, dyQ) * 0.25f;
            
            sad = 0.0f;
            for (int byQ = -BLOCK_R; byQ <= BLOCK_R; ++byQ) {
                for (int bxQ = -BLOCK_R; bxQ <= BLOCK_R; ++bxQ) {
                     int2 sp = localPos + int2(bxQ, byQ);
                     float cVal = gs_Curr[sp.y][sp.x];
                     
                     float2 pPos = float2(pos) + float2(bxQ, byQ) + 0.5f + testMV;
                     float2 pUV = pPos / float2(w, h);
                     sad += abs(cVal - PrevLuma.SampleLevel(LinearClamp, pUV, 0));
                }
            }
            
            if (sad < bestQuarterSad) {
                bestQuarterSad = sad;
                bestQuarterMV = testMV;
            }
        }
    }
    
    // Confidence
    float avgDiff = bestQuarterSad / float((2 * BLOCK_R + 1) * (2 * BLOCK_R + 1));
    float confidence = exp(-avgDiff * 10.0f);
    confidence *= smoothstep(0.0f, 0.2f, frameDiff);
    confidence = clamp(confidence, 0.15f, 0.98f);
    
    if (length(bestQuarterMV) < 0.5f && frameDiff < 0.025f) {
        confidence = max(confidence, 0.88f);
    }
    
    MotionOut[id.xy] = bestQuarterMV;
    ConfidenceOut[id.xy] = confidence;
}

