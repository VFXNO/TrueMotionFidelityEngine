// ============================================================================
// MOTION SMOOTH v2 - Bilateral Filter (AI-Like Structure Awareness)
// 5x5 window with Spatial, Color (Luma), and Confidence weights.
// ============================================================================

Texture2D<float2> MotionIn : register(t0);
Texture2D<float> ConfIn : register(t1);
Texture2D<float> LumaIn : register(t2);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfOut : register(u1);

cbuffer SmoothCB : register(b0) {
    float edgeScale;
    float confPower;
    float2 pad;
};

// Controls the Gaussian falloff for spatial distance.
// Specified as denominator D in exp(-dist^2 / D).
// Value ~2.0 keeps immediate neighbors strong, corners weaker.
static const float SIGMA_S = 2.0;

// Base denominator for color difference.
// exp(-diff^2 / sigmaR).
// Value ~0.1 makes it sensitive to edges (diff > 0.3 becomes negligible).
static const float BASE_SIGMA_R = 0.1;

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    MotionIn.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int2 centerPos = int2(id.xy);

    // Modulate Sigma Color by Edge Scale.
    // If edgeScale is high, we allow more blurring (higher sigma).
    // If edgeScale is low, we preserve edges strictly.
    // We clamp minimum to avoid division by zero or extreme sensitivity.
    float sigmaR = BASE_SIGMA_R * max(0.01, edgeScale);

    // Center Data for Comparisons
    float centerLuma = LumaIn.Load(int3(centerPos, 0));
    
    // Accumulators
    float2 sumMotion = float2(0, 0);
    float sumConf = 0.0;
    float totalWeight = 0.0;

    // 5x5 Window
    // Loop y: -2 to +2, x: -2 to +2
    [unroll]
    for (int y = -2; y <= 2; ++y)
    {
        [unroll]
        for (int x = -2; x <= 2; ++x)
        {
            int2 offset = int2(x, y);
            int2 samplePos = centerPos + offset;
            
            // Bounds Check / Clamping
            // Critical for sampling edges of screen correctly
            samplePos = clamp(samplePos, int2(0, 0), int2(w - 1, h - 1));

            // Load Neighbor Properties
            float2 nMotion = MotionIn.Load(int3(samplePos, 0));
            float nConf = ConfIn.Load(int3(samplePos, 0)); // Confidence 0..1
            float nLuma = LumaIn.Load(int3(samplePos, 0));

            // 1. Spatial Weight: exp(-dist^2 / sigmaS)
            // Distance squared from center (0,0) of the window
            float distSq = dot(float2(offset), float2(offset));
            float wSpatial = exp(-distSq / SIGMA_S);

            // 2. Color Weight: exp(-diff^2 / sigmaR)
            // Bilateral component: prevents smoothing across luma edges
            float lumaDiff = centerLuma - nLuma;
            float wColor = exp(-(lumaDiff * lumaDiff) / sigmaR);

            // 3. Confidence Weight
            // Multiply by neighbor's confidence. 
            // Neighbors with 0 confidence contribute 0 to the sum.
            // This naturally fills holes (low conf pixels) with data from valid neighbors.
            float wConf = nConf;
            // (Optional) Apply power curve from CB if needed: pow(nConf, confPower)
            // but linear multiplication is robust and stable.

            // Combine weights
            float weight = wSpatial * wColor * wConf;

            // Accumulate
            sumMotion += nMotion * weight;
            sumConf += nConf * weight; 
            totalWeight += weight;
        }
    }

    // Normalize and Output
    float2 finalMotion;
    float finalConf;

    // If we gathered valid data (weight > 0), normalize.
    if (totalWeight > 1e-6)
    {
        finalMotion = sumMotion / totalWeight;
        finalConf = sumConf / totalWeight;
    }
    else
    {
        // Fallback: If area is completely invalid (all 0 confidence)
        // or weights excessively suppressed, pass through center.
        finalMotion = MotionIn.Load(int3(centerPos, 0));
        finalConf = ConfIn.Load(int3(centerPos, 0));
    }

    MotionOut[id.xy] = finalMotion;
    ConfOut[id.xy] = finalConf;
}
