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
    float confPower;   // Unused, kept for API
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
    
    // Adaptive sigma based on edge strength
    float baseSigma = 1.5;
    float sigmaMV = 2.5;  // Motion similarity sigma
    float sigmaLuma = 0.08 * max(0.5, edgeScale);
    
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
            
            // Confidence weight
            float wc = 0.3 + 0.7 * conf;
            
            float weight = ws * wl * wm * wc;
            sumMV += mv * weight;
            sumConf += conf * weight;
            sumW += weight;
        }
    }
    
    // Output
    if (sumW > 0.001) {
        MotionOut[id.xy] = sumMV / sumW;
        ConfOut[id.xy] = sumConf / sumW;
    } else {
        MotionOut[id.xy] = centerMV;
        ConfOut[id.xy] = centerConf;
    }
}
