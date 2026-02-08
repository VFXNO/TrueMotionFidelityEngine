// ============================================================================
// MOTION REFINEMENT - Professional Sub-Pixel Accuracy
// Gradient-weighted refinement with equiangular sub-pixel fitting
// Optimized using Shared Memory and Single-Pass ZNSSD
// ============================================================================

// C++ binds: {CurrLuma, PrevLuma, CoarseMotion, CoarseConf}
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
    float2 pad;
};

#define BLOCK_SIZE 16
#define MAX_RADIUS 4
#define WINDOW_R 3
// Halo required: Radius + Window = 4 + 3 = 7
#define MARGIN 7
#define SHARED_DIM (BLOCK_SIZE + 2 * MARGIN) // 30 -> 32 for alignment

groupshared float gs_Luma[SHARED_DIM][SHARED_DIM];

// ============================================================================
// Bilinear Luma Sample (for PrevLuma)
// ============================================================================
float SamplePrev(float2 pos, float2 size) {
    float2 uv = (pos + 0.5) / size;
    return PrevLuma.SampleLevel(LinearClamp, uv, 0);
}

// ============================================================================
// 7x7 Gaussian weights (Pre-flattened for access)
// ============================================================================
static const float kGaussian[49] = {
    0.005, 0.012, 0.020, 0.024, 0.020, 0.012, 0.005,
    0.012, 0.027, 0.044, 0.052, 0.044, 0.027, 0.012,
    0.020, 0.044, 0.072, 0.085, 0.072, 0.044, 0.020,
    0.024, 0.052, 0.085, 0.100, 0.085, 0.052, 0.024,
    0.020, 0.044, 0.072, 0.085, 0.072, 0.044, 0.020,
    0.012, 0.027, 0.044, 0.052, 0.044, 0.027, 0.012,
    0.005, 0.012, 0.020, 0.024, 0.020, 0.012, 0.005
};

// ============================================================================
// Helper to get cached luma from shared memory
// ============================================================================
float GetCachedLuma(int2 localPos) {
    return gs_Luma[localPos.y][localPos.x];
}

// ============================================================================
// Compute Gradient Weight from Shared Memory
// ============================================================================
float GetCachedGradientWeight(int2 localPos) {
    float l = gs_Luma[localPos.y][localPos.x - 1];
    float r = gs_Luma[localPos.y][localPos.x + 1];
    float u = gs_Luma[localPos.y - 1][localPos.x];
    float d = gs_Luma[localPos.y + 1][localPos.x];
    
    float gx = r - l;
    float gy = d - u;
    // 1 + 4 * mag
    return 1.0 + sqrt(gx*gx + gy*gy) * 4.0;
}

// ============================================================================
// Optimized Single-Pass ZNSSD Cost
// Uses precomputed Curr Mean/Var terms passed in arguments
// ============================================================================
float ComputeCostOptimized(
    int2 localCenter, 
    float2 mvec, 
    uint2 dims, 
    int2 globalPos,
    float meanCurr, 
    float wVarSumCurr,
    float sumW) 
{
    float sumPrev = 0.0;
    float sumPrev2 = 0.0;
    float sumCP = 0.0;
    
    float2 size = float2(dims);
    
    // Iterate 7x7 window
    [loop] // Use loop to reduce register pressure
    for (int dy = -WINDOW_R; dy <= WINDOW_R; ++dy) {
        [loop]
        for (int dx = -WINDOW_R; dx <= WINDOW_R; ++dx) {
            int idx = (dy + 3) * 7 + (dx + 3);
            float sW = kGaussian[idx];
            
            int2 lPos = localCenter + int2(dx, dy);
            
            // Recompute gradient weight from shared memory (fast L1)
            float gW = GetCachedGradientWeight(lPos);
            float W = sW * gW;
            
            // Get cached Curr value
            float cVal = GetCachedLuma(lPos);
            
            // Sample Prev value (global memory)
            // Note: globalPos is the center pixel. 
            // Position to sample = globalPos + offset + mvec
            float2 pUVPos = float2(globalPos) + float2(dx, dy) + mvec;
            float pVal = SamplePrev(pUVPos, size);
            
            sumPrev += pVal * W;
            sumPrev2 += pVal * pVal * W;
            sumCP += cVal * pVal * W;
        }
    }
    
    // ZNSSD Calculation
    // SSD = Sum(W * ((C - mC) - (P - mP))^2)
    //     = VarSumC + VarSumP - 2 * CovSumCP
    // VarSumP = Sum(W * P^2) - (Sum(W*P)^2 / SumW)
    // CovSumCP = Sum(W * C * P) - (Sum(W*C)*Sum(W*P) / SumW)
    
    // Note: meanCurr = Sum(W*C)/SumW
    // So CovSumCP = SumCP - meanCurr * SumPrev
    
    float meanPrev = sumPrev / max(sumW, 0.001);
    float wVarSumPrev = sumPrev2 - (sumPrev * sumPrev) / max(sumW, 0.001);
    float wCovSum = sumCP - (meanCurr * sumPrev);
    
    float ssd = max(0.0, wVarSumCurr + wVarSumPrev - 2.0 * wCovSum);
    
    // StdDev product for normalization
    float stdDev = sqrt(max(wVarSumCurr, 0.0001) * max(wVarSumPrev, 0.0001));
    
    return sqrt(ssd / max(sumW, 0.001)) / max(stdDev, 0.05);
}

// ============================================================================
// MAIN
// ============================================================================
[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint w, h;
    CurrLuma.GetDimensions(w, h);
    
    // ------------------------------------------------------------------------
    // 1. Cooperative Load into Shared Memory
    // ------------------------------------------------------------------------
    int2 groupBase = gid.xy * BLOCK_SIZE - MARGIN;
    uint tid = gtid.y * BLOCK_SIZE + gtid.x; // 0..255
    
    // Total shared pixels = 32*32 = 1024
    // Each thread loads 4 pixels
    for (int i = 0; i < 4; ++i) {
        uint pIdx = tid + i * 256;
        if (pIdx < SHARED_DIM * SHARED_DIM) {
            int ly = pIdx / SHARED_DIM;
            int lx = pIdx % SHARED_DIM;
            int2 loadPos = groupBase + int2(lx, ly);
            
            // Clamp to border
            loadPos = clamp(loadPos, int2(0,0), int2(w-1, h-1));
            gs_Luma[ly][lx] = CurrLuma.Load(int3(loadPos, 0));
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    if (id.x >= w || id.y >= h) return;
    
    // Local center in shared memory
    int2 localCenter = int2(gtid.xy) + MARGIN;
    int2 globalPos = int2(id.xy);

    // ------------------------------------------------------------------------
    // 2. Precompute Current Block Statistics (Mean, VarSum)
    // ------------------------------------------------------------------------
    float sumC = 0, sumC2 = 0, sumW = 0;
    
    [unroll]
    for (int dy = -WINDOW_R; dy <= WINDOW_R; ++dy) {
        [unroll]
        for (int dx = -WINDOW_R; dx <= WINDOW_R; ++dx) {
            int idx = (dy + 3) * 7 + (dx + 3);
            int2 lPos = localCenter + int2(dx, dy);
            
            float sW = kGaussian[idx];
            float gW = GetCachedGradientWeight(lPos);
            float W = sW * gW;
            float cVal = GetCachedLuma(lPos);
            
            sumC += cVal * W;
            sumC2 += cVal * cVal * W;
            sumW += W;
        }
    }
    
    float meanCurr = sumC / max(sumW, 0.001);
    float wVarSumCurr = sumC2 - (sumC * sumC) / max(sumW, 0.001);
    
    // ------------------------------------------------------------------------
    // 3. Coarse Search
    // ------------------------------------------------------------------------
    float2 uv = (float2(id.xy) + 0.5) / float2(w, h);
    float2 coarse = CoarseMotion.SampleLevel(LinearClamp, uv, 0);
    
    // Keep fractional prediction from bilinear upsample so the 8x8 flow grid
    // is expanded into per-pixel motion instead of snapping to integer cells.
    float2 pred = coarse * motionScale;
    float2 baseMV = pred;
    
    float bestCost = 1e9;
    float2 bestMV = baseMV;
    int searchR = clamp(radius, 1, MAX_RADIUS);
    
    for (int sy = -searchR; sy <= searchR; ++sy) {
        for (int sx = -searchR; sx <= searchR; ++sx) {
            float2 testMV = baseMV + float2(sx, sy);
            
            float cost = ComputeCostOptimized(
                localCenter, testMV, uint2(w,h), globalPos,
                meanCurr, wVarSumCurr, sumW
            );
            
            // Regularization
            cost += length(float2(sx, sy)) * 0.01;
            
            if (cost < bestCost) {
                bestCost = cost;
                bestMV = testMV;
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // 4. Sub-Pixel Refinement (Quadratic Fit)
    // ------------------------------------------------------------------------
    // Optimized: Only fit quadratic around the best local match
    // Instead of 8-direction search which is expensive
    // Just calculate +/- neighbours and fit
    
    float2 finalMV = bestMV;
    
    // X-Refinement
    float cL = ComputeCostOptimized(localCenter, finalMV + float2(-1, 0), uint2(w, h), globalPos, meanCurr, wVarSumCurr, sumW);
    float cR = ComputeCostOptimized(localCenter, finalMV + float2( 1, 0), uint2(w, h), globalPos, meanCurr, wVarSumCurr, sumW);
    float cC = bestCost;
    
    float denomX = cL + cR - 2.0 * cC;
    float subX = (abs(denomX) > 1e-5) ? (0.5 * (cL - cR) / denomX) : 0.0;
    
    // Y-Refinement
    float cU = ComputeCostOptimized(localCenter, finalMV + float2(0, -1), uint2(w, h), globalPos, meanCurr, wVarSumCurr, sumW);
    float cD = ComputeCostOptimized(localCenter, finalMV + float2(0,  1), uint2(w, h), globalPos, meanCurr, wVarSumCurr, sumW);
    
    float denomY = cU + cD - 2.0 * cC;
    float subY = (abs(denomY) > 1e-5) ? (0.5 * (cU - cD) / denomY) : 0.0;
    
    finalMV += float2(clamp(subX, -0.5, 0.5), clamp(subY, -0.5, 0.5));
    
    // Final Confidence
    float diff = bestCost; // Use min cost as proxy for inverse confidence
    float conf = exp(-diff * 4.0);
    conf = clamp(conf, 0.1, 0.98);
    
    MotionOut[id.xy] = finalMV;
    ConfidenceOut[id.xy] = conf;
}
