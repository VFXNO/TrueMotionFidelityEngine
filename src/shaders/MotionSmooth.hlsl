// ============================================================================
// MOTION SMOOTHING - Structure-Tensor Anisotropic Bilateral Filter
// Edge-preserving smoothing that follows image structure
// ============================================================================

Texture2D<float2> MotionIn : register(t0);
Texture2D<float> ConfIn : register(t1);
Texture2D<float> LumaIn : register(t2);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfOut : register(u1);

cbuffer SmoothCB : register(b0) {
    float edgeScale;   // Edge preservation strength
    float confPower;   // Confidence tightening for vector filtering
    float2 pad;
};

// ============================================================================
// Structure Tensor: Computes local edge orientation
// Returns principal direction and anisotropy measure
// ============================================================================
void ComputeStructureTensor(int2 pos, uint w, uint h, out float2 direction, out float anisotropy)
{
    // Sobel gradients in 3x3 neighborhood
    float gxx = 0, gyy = 0, gxy = 0;
    
    // [unroll] removed
    for (int dy = -1; dy <= 1; ++dy) {
        // [unroll] removed
        for (int dx = -1; dx <= 1; ++dx) {
            int2 p = clamp(pos + int2(dx, dy), int2(0,0), int2(w-1, h-1));
            
            float l = LumaIn.Load(int3(clamp(p + int2(-1, 0), int2(0,0), int2(w-1,h-1)), 0));
            float r = LumaIn.Load(int3(clamp(p + int2( 1, 0), int2(0,0), int2(w-1,h-1)), 0));
            float u = LumaIn.Load(int3(clamp(p + int2( 0,-1), int2(0,0), int2(w-1,h-1)), 0));
            float d = LumaIn.Load(int3(clamp(p + int2( 0, 1), int2(0,0), int2(w-1,h-1)), 0));
            
            float gx = (r - l) * 0.5;
            float gy = (d - u) * 0.5;
            
            gxx += gx * gx;
            gyy += gy * gy;
            gxy += gx * gy;
        }
    }
    
    // Eigenvalue decomposition of 2x2 structure tensor
    float trace = gxx + gyy;
    float det = gxx * gyy - gxy * gxy;
    float disc = sqrt(max(trace * trace * 0.25 - det, 0.0));
    
    float lambda1 = trace * 0.5 + disc;  // Larger eigenvalue
    float lambda2 = trace * 0.5 - disc;  // Smaller eigenvalue
    
    // Anisotropy: 0 = isotropic (no edge), 1 = strong edge
    anisotropy = (lambda1 > 0.0001) ? (lambda1 - lambda2) / (lambda1 + lambda2 + 0.0001) : 0.0;
    
    // Principal direction (perpendicular to gradient = along edge)
    if (abs(gxy) > 0.0001) {
        direction = normalize(float2(lambda1 - gxx, gxy));
    } else {
        direction = (gxx > gyy) ? float2(0, 1) : float2(1, 0);
    }
}

// ============================================================================
// MAIN
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    MotionIn.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int2 pos = int2(id.xy);
    
    // Center data
    float2 centerMV = MotionIn.Load(int3(pos, 0));
    float centerConf = ConfIn.Load(int3(pos, 0));
    float centerLuma = LumaIn.Load(int3(pos, 0));
    
    // Compute structure tensor for anisotropic filtering
    float2 edgeDir;
    float anisotropy;
    ComputeStructureTensor(pos, w, h, edgeDir, anisotropy);
    
    // Adaptive sigma based on edge strength.
    // Smaller sigma at higher edgeScale prevents vectors leaking across boundaries.
    float baseSigma = 1.5;
    float sigmaMV = lerp(2.5, 1.4, anisotropy);  // Tighter near strong edges
    float sigmaLuma = clamp(0.08 / max(0.5, edgeScale), 0.01, 0.12);
    float confTight = saturate((confPower - 0.25) / 3.75);
    
    float2 sumMV = float2(0, 0);
    float sumConf = 0.0;
    float sumW = 0.0;
    
    // 5x5 anisotropic bilateral filter
    // [unroll] removed
    for (int dy = -2; dy <= 2; ++dy) {
        // [unroll] removed
        for (int dx = -2; dx <= 2; ++dx) {
            int2 sp = clamp(pos + int2(dx, dy), int2(0,0), int2(w-1, h-1));
            
            float2 mv = MotionIn.Load(int3(sp, 0));
            float conf = ConfIn.Load(int3(sp, 0));
            float luma = LumaIn.Load(int3(sp, 0));
            
            // Offset vector
            float2 offset = float2(dx, dy);
            
            // Anisotropic spatial weight
            // Project offset onto edge direction and perpendicular
            float alongEdge = dot(offset, edgeDir);
            float acrossEdge = length(offset - alongEdge * edgeDir);
            
            // Stretch kernel along edge direction
            float stretchFactor = lerp(1.0, 3.0, anisotropy);
            float effectiveDist = sqrt(alongEdge * alongEdge / (stretchFactor * stretchFactor) + acrossEdge * acrossEdge);
            float ws = exp(-effectiveDist * effectiveDist / (2.0 * baseSigma * baseSigma));
            
            // Luma (edge) weight
            float lumaDiff = abs(luma - centerLuma);
            float wl = exp(-lumaDiff / sigmaLuma);
            
            // Motion similarity weight
            float2 mvDiff = mv - centerMV;
            float mvDist2 = dot(mvDiff, mvDiff);
            float wm = exp(-mvDist2 / (sigmaMV * sigmaMV));
            
            // Confidence-aware weight: favor reliable neighbors and keep center ownership.
            float wcBase = 0.2 + 0.8 * conf;
            float wcTight = conf * conf;
            float wc = lerp(wcBase, wcTight, confTight);
            wc *= (0.5 + 0.5 * centerConf);
            float confAgreement = 1.0 - abs(conf - centerConf);
            wc *= saturate(0.4 + 0.6 * confAgreement);
            wc = max(wc, 0.05);
            
            float weight = ws * wl * wm * wc;
            sumMV += mv * weight;
            sumConf += conf * weight;
            sumW += weight;
        }
    }
    
    float2 firstMV = centerMV;
    float firstConf = centerConf;
    if (sumW > 0.001) {
        firstMV = sumMV / sumW;
        firstConf = sumConf / sumW;
    }

    // Robust second stage: suppress outlier vectors that survive bilateral weights.
    float2 sumMV2 = float2(0, 0);
    float sumConf2 = 0.0;
    float sumW2 = 0.0;
    float sigmaRobust = lerp(1.6, 0.9, anisotropy);
    float sigmaRobust2 = sigmaRobust * sigmaRobust;

    for (int dy2 = -2; dy2 <= 2; ++dy2) {
        for (int dx2 = -2; dx2 <= 2; ++dx2) {
            int2 sp = clamp(pos + int2(dx2, dy2), int2(0,0), int2(w-1, h-1));

            float2 mv = MotionIn.Load(int3(sp, 0));
            float conf = ConfIn.Load(int3(sp, 0));
            float luma = LumaIn.Load(int3(sp, 0));

            float2 offset = float2(dx2, dy2);
            float alongEdge = dot(offset, edgeDir);
            float acrossEdge = length(offset - alongEdge * edgeDir);
            float stretchFactor = lerp(1.0, 3.0, anisotropy);
            float effectiveDist = sqrt(alongEdge * alongEdge / (stretchFactor * stretchFactor) + acrossEdge * acrossEdge);
            float ws = exp(-effectiveDist * effectiveDist / (2.0 * baseSigma * baseSigma));

            float lumaDiff = abs(luma - centerLuma);
            float wl = exp(-lumaDiff / sigmaLuma);

            float2 mvDiff = mv - firstMV;
            float mvDist2 = dot(mvDiff, mvDiff);
            float wm = exp(-mvDist2 / (sigmaMV * sigmaMV));
            float wr = rcp(1.0 + mvDist2 / max(0.25, sigmaRobust2));

            float wcBase = 0.2 + 0.8 * conf;
            float wcTight = conf * conf;
            float wc = lerp(wcBase, wcTight, confTight);
            wc *= (0.5 + 0.5 * firstConf);
            float confAgreement = 1.0 - abs(conf - firstConf);
            wc *= saturate(0.4 + 0.6 * confAgreement);
            wc = max(wc, 0.05);

            float weight = ws * wl * wm * wr * wc;
            sumMV2 += mv * weight;
            sumConf2 += conf * weight;
            sumW2 += weight;
        }
    }

    if (sumW2 > 0.001) {
        MotionOut[id.xy] = sumMV2 / sumW2;
        ConfOut[id.xy] = sumConf2 / sumW2;
    } else if (sumW > 0.001) {
        MotionOut[id.xy] = firstMV;
        ConfOut[id.xy] = firstConf;
    } else {
        MotionOut[id.xy] = centerMV;
        ConfOut[id.xy] = centerConf;
    }
}

