// ============================================================================
// DOWNSAMPLE LUMA & TINY CNN FEATURE EXTRACTION
// RGB to Luma with 2x2 box downsample + AI-based feature extraction
// Extracts: [Luma, EdgeX, EdgeY, TexturePattern] for robust motion estimation
// ============================================================================

Texture2D<float4> Src : register(t0);
RWTexture2D<float4> LumaOut : register(u0);
RWTexture2D<float4> Feature2Out : register(u1);
RWTexture2D<float4> Feature3Out : register(u2);

static const float3 kLumaWeights = float3(0.2126, 0.7152, 0.0722);

// saturate for older shader models
float saturate(float x) { return clamp(x, 0.0, 1.0); }

float GetLuma(int2 pos, int2 maxPos) {
    pos = clamp(pos, int2(0, 0), maxPos);
    return dot(Src.Load(int3(pos, 0)).rgb, kLumaWeights);
}

float GetAvgLuma(int2 base, int2 maxPos) {
    float l00 = GetLuma(base, maxPos);
    float l10 = GetLuma(base + int2(1, 0), maxPos);
    float l01 = GetLuma(base + int2(0, 1), maxPos);
    float l11 = GetLuma(base + int2(1, 1), maxPos);
    return (l00 + l10 + l01 + l11) * 0.25;
}

// ============================================================================
// Walsh-Hadamard Transform (WHT) for Periodicity Detection
// Detects repetitive/periodic texture patterns that confuse optical flow
// ============================================================================

float ComputePeriodicityWHT(int2 base, int2 maxPos) {
    // Sample 4x4 neighborhood for WHT
    float s[4][4];
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            s[y][x] = GetAvgLuma(base + int2(x * 2 - 3, y * 2 - 3), maxPos);
        }
    }
    
    // 4x4 WHT - no multiplications, only +1/-1
    // H4 = H2 ⊗ H2 where H2 = [[1,1],[1,-1]]
    float wht[4][4];
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            float sum = 0;
            for (int ky = 0; ky < 4; ky++) {
                for (int kx = 0; kx < 4; kx++) {
                    int sign = ((ky & 1) ? -1 : 1) * ((kx & 1) ? -1 : 1);
                    sum += s[ky][kx] * sign;
                }
            }
            wht[y][x] = sum * 0.25;
        }
    }
    
    // Compute DC (mean) and AC energy
    float dc = wht[0][0];
    float acEnergy = 0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            if (y != 0 || x != 0) {
                acEnergy += wht[y][x] * wht[y][x];
            }
        }
    }
    acEnergy = sqrt(acEnergy / 15.0);
    
    // Periodicity metric: high AC energy concentrated in few bins = periodic
    // If AC is spread uniformly = random texture (not periodic)
    // We check if there are strong peaks
    
    // Find max AC coefficient
    float maxAC = 0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            if (y != 0 || x != 0) {
                maxAC = max(maxAC, abs(wht[y][x]));
            }
        }
    }
    
    // If max AC is much larger than RMS AC → periodic pattern
    float rmsAC = sqrt(acEnergy * acEnergy + 1e-10);
    float peakRatio = maxAC / (rmsAC + 1e-10);
    
    // Combined periodicity score (0 = no periodicity, 1 = highly periodic)
    // Peak ratio > 2.0 suggests strong periodicity
    float periodicity = saturate(peakRatio - 1.5) * 0.5;
    
    // Also check for checkerboard-like patterns (alternating)
    float checker = abs(s[0][0] - s[1][1]) + abs(s[1][0] - s[0][1]);
    float variance = 0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            variance += abs(s[y][x] - dc);
        }
    }
    variance /= 16.0;
    
    // If checkerboard energy is high relative to variance → periodic
    float checkerboardness = saturate(checker / (variance + 0.01) - 0.5) * 0.3;
    
    return min(periodicity + checkerboardness, 1.0);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint outW, outH;
    LumaOut.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;

    uint inW, inH;
    Src.GetDimensions(inW, inH);
    int2 maxPos = int2(inW - 1, inH - 1);
    int2 base = int2(id.xy * 2);

    // Sample 3x3 neighborhood of the downsampled luma
    float p00 = GetAvgLuma(base + int2(-2, -2), maxPos);
    float p10 = GetAvgLuma(base + int2( 0, -2), maxPos);
    float p20 = GetAvgLuma(base + int2( 2, -2), maxPos);
    
    float p01 = GetAvgLuma(base + int2(-2,  0), maxPos);
    float p11 = GetAvgLuma(base + int2( 0,  0), maxPos); // Center
    float p21 = GetAvgLuma(base + int2( 2,  0), maxPos);
    
    float p02 = GetAvgLuma(base + int2(-2,  2), maxPos);
    float p12 = GetAvgLuma(base + int2( 0,  2), maxPos);
    float p22 = GetAvgLuma(base + int2( 2,  2), maxPos);

    // WHT-based Periodicity Detection (repetitive texture indicator)
    float f_periodic = ComputePeriodicityWHT(base, maxPos);

    // Tiny CNN Layer 1: Feature Extraction (Hand-crafted weights)
    // Feature 1: Base Luma
    float f_luma = p11;
    
    // Feature 2 & 3: Scharr Operator (Better rotational symmetry than Sobel)
    // Scharr weights (3, 10, 3) detect edges at odd angles much more accurately than Sobel.
    float f_edgeX = ((3.0*p20 + 10.0*p21 + 3.0*p22) - (3.0*p00 + 10.0*p01 + 3.0*p02)) * 0.25;
    float f_edgeY = ((3.0*p02 + 10.0*p12 + 3.0*p22) - (3.0*p00 + 10.0*p10 + 3.0*p20)) * 0.25;
    
    // Feature 4: Texture Pattern (Difference of Gaussians / High-Pass)
    // DoG is more robust to noise than a simple mean, acting like a SIFT feature detector.
    float blur = (p00+p02+p20+p22)*0.0625 + (p01+p10+p12+p21)*0.125 + p11*0.25;
    float f_tex = (p11 - blur) * 5.0;

    // Feature 5: Corner Response (Harris-like)
    // Corners are the most reliable features for optical flow because they don't suffer from the aperture problem.
    float ixx = f_edgeX * f_edgeX;
    float iyy = f_edgeY * f_edgeY;
    float ixy = f_edgeX * f_edgeY;
    float f_corner = (ixx * iyy - ixy * ixy) - 0.05 * (ixx + iyy) * (ixx + iyy);
    f_corner *= 5.0; // Scale up for visibility

    // Feature 6: Local Variance (Texture Energy)
    float mean = (p00+p10+p20+p01+p11+p21+p02+p12+p22) / 9.0;
    float var = ((p00-mean)*(p00-mean) + (p10-mean)*(p10-mean) + (p20-mean)*(p20-mean) +
                 (p01-mean)*(p01-mean) + (p11-mean)*(p11-mean) + (p21-mean)*(p21-mean) +
                 (p02-mean)*(p02-mean) + (p12-mean)*(p12-mean) + (p22-mean)*(p22-mean)) / 9.0;
    float f_var = sqrt(var) * 2.0;

    // Feature 7 & 8: Diagonal Gradients
    float f_diag1 = (p22 - p00) * 2.0;
    float f_diag2 = (p20 - p02) * 2.0;

    // Feature 9: Smoothed Luma (Low-pass filter for flat areas)
    float f_smooth = blur;

    // Feature 10: Laplacian of Gaussian (LoG) - Band-pass filter for blob detection
    float f_log = (p10 + p01 + p21 + p12) - 4.0 * p11;

    // Feature 11: Edge Magnitude (Rotation invariant edge strength)
    float f_mag = sqrt(ixx + iyy);

    // Feature 12: Cross Derivative (Saddle point detection)
    float f_cross = ixy;

    // Advanced CNN Layer 2: Fast Activation (Softsign)
    // Replaced expensive exp() with a highly optimized Softsign activation: x / (1.0 + abs(x))
    // This gives the exact same non-linear thresholding benefits but at a fraction of the GPU cost.
    float beta = 2.0;
    f_edgeX = f_edgeX / (1.0 + beta * abs(f_edgeX));
    f_edgeY = f_edgeY / (1.0 + beta * abs(f_edgeY));
    f_diag1 = f_diag1 / (1.0 + beta * abs(f_diag1));
    f_diag2 = f_diag2 / (1.0 + beta * abs(f_diag2));
    f_corner = sign(f_corner) * (abs(f_corner) / (1.0 + abs(f_corner)));
    f_log = f_log / (1.0 + beta * abs(f_log));
    f_cross = f_cross / (1.0 + beta * abs(f_cross));
    // f_tex, f_var, f_smooth, and f_mag are passed linearly to preserve the exact texture pattern for optical flow

    // Output the 12-channel feature map
    // WHT periodicity: 0 = random/noise, 1 = highly repetitive texture
    LumaOut[id.xy] = float4(f_luma, f_edgeX, f_edgeY, f_tex);
    Feature2Out[id.xy] = float4(f_corner, f_var, f_diag1, f_diag2);
    Feature3Out[id.xy] = float4(f_smooth, f_log, f_mag, f_periodic);
}
