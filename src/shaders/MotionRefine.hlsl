// ============================================================================
// MOTION REFINEMENT - Sub-pixel SAD Refinement
// Quarter-pixel accuracy for smooth motion
// ============================================================================

Texture2D<float> CurrLuma : register(t0);
Texture2D<float> PrevLuma : register(t1);
Texture2D<float2> CoarseMotion : register(t2);
Texture2D<float> CoarseConf : register(t3);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfidenceOut : register(u1);

SamplerState LinearClamp : register(s0);

cbuffer RefineCB : register(b0) {
    int radius;
    float motionScale;
    float pad[2];
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    CurrLuma.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    
    int2 pos = int2(id.xy);
    float2 uv = (float2(pos) + 0.5) / float2(w, h);
    
    float2 coarseMV = CoarseMotion.SampleLevel(LinearClamp, uv, 0).xy * motionScale;
    float coarseConf = CoarseConf.SampleLevel(LinearClamp, uv, 0).x;
    
    int2 baseMV = int2(round(coarseMV));
    int searchR = clamp(radius, 1, 4);
    
    float bestSad = 1e9f;
    float2 bestMV = coarseMV;
    
    for (int dy = -searchR; dy <= searchR; ++dy) {
        for (int dx = -searchR; dx <= searchR; ++dx) {
            int2 testMV = baseMV + int2(dx, dy);
            
            float sad = 0.0f;
            for (int by = -3; by <= 3; ++by) {
                for (int bx = -3; bx <= 3; ++bx) {
                    int2 cPos = clamp(pos + int2(bx, by), int2(0, 0), int2(w - 1, h - 1));
                    int2 pPos = clamp(pos + int2(bx, by) + testMV, int2(0, 0), int2(w - 1, h - 1));
                    sad += abs(CurrLuma.Load(int3(cPos, 0)) - PrevLuma.Load(int3(pPos, 0)));
                }
            }
            
            if (sad < bestSad) {
                bestSad = sad;
                bestMV = float2(testMV);
            }
        }
    }
    
    float2 bestHalfMV = bestMV;
    float bestHalfSad = bestSad;
    
    for (int dyH = -1; dyH <= 1; ++dyH) {
        for (int dxH = -1; dxH <= 1; ++dxH) {
            if (dxH == 0 && dyH == 0) continue;
            
            float2 testMV = bestMV + float2(dxH, dyH) * 0.5f;
            
            float sad = 0.0f;
            for (int byH = -3; byH <= 3; ++byH) {
                for (int bxH = -3; bxH <= 3; ++bxH) {
                    float2 cPos = float2(pos) + float2(bxH, byH) + 0.5f;
                    float2 pPos = cPos + testMV;
                    float2 pUV = pPos / float2(w, h);
                    pUV = clamp(pUV, 0.0f, 0.999f);
                    sad += abs(CurrLuma.Load(int3(cPos, 0)) - PrevLuma.SampleLevel(LinearClamp, pUV, 0));
                }
            }
            
            if (sad < bestHalfSad) {
                bestHalfSad = sad;
                bestHalfMV = testMV;
            }
        }
    }
    
    float2 bestQuarterMV = bestHalfMV;
    float bestQuarterSad = bestHalfSad;
    
    for (int dyQ = -1; dyQ <= 1; ++dyQ) {
        for (int dxQ = -1; dxQ <= 1; ++dxQ) {
            if (dxQ == 0 && dyQ == 0) continue;
            
            float2 testMV = bestHalfMV + float2(dxQ, dyQ) * 0.25f;
            
            float sad = 0.0f;
            for (int byQ = -3; byQ <= 3; ++byQ) {
                for (int bxQ = -3; bxQ <= 3; ++bxQ) {
                    float2 cPos = float2(pos) + float2(bxQ, byQ) + 0.5f;
                    float2 pPos = cPos + testMV;
                    float2 pUV = pPos / float2(w, h);
                    pUV = clamp(pUV, 0.0f, 0.999f);
                    sad += abs(CurrLuma.Load(int3(cPos, 0)) - PrevLuma.SampleLevel(LinearClamp, pUV, 0));
                }
            }
            
            if (sad < bestQuarterSad) {
                bestQuarterSad = sad;
                bestQuarterMV = testMV;
            }
        }
    }
    
    float confidence = exp(-bestQuarterSad * 0.1f / 49.0f);
    confidence = lerp(confidence, coarseConf, 0.3f);
    confidence = clamp(confidence, 0.1f, 0.98f);
    
    MotionOut[id.xy] = bestQuarterMV;
    ConfidenceOut[id.xy] = confidence;
}
