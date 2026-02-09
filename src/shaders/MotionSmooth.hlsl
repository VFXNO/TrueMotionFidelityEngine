// ============================================================================
// MOTION SMOOTHING - Light Edge-Preserving Filter
// Small kernel for minimal smoothing
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

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    MotionIn.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int2 pos = int2(id.xy);
    
    float2 centerMV = MotionIn.Load(int3(pos, 0));
    float centerConf = ConfIn.Load(int3(pos, 0));
    float centerLuma = LumaIn.Load(int3(pos, 0));
    
    float lL = LumaIn.Load(int3(clamp(pos + int2(-1, 0), int2(0, 0), int2(w - 1, h - 1)), 0));
    float lR = LumaIn.Load(int3(clamp(pos + int2(1, 0), int2(0, 0), int2(w - 1, h - 1)), 0));
    float lU = LumaIn.Load(int3(clamp(pos + int2(0, -1), int2(0, 0), int2(w - 1, h - 1)), 0));
    float lD = LumaIn.Load(int3(clamp(pos + int2(0, 1), int2(0, 0), int2(w - 1, h - 1)), 0));
    
    float edgeStrength = abs(lR - lL) + abs(lU - lD);
    
    float2 sumMV = float2(0, 0);
    float sumConf = 0.0;
    float sumW = 0.0;
    
    // Optimized: 5x5 Kernel (-2..2) - Sufficient for smoothing, 9x9 was redundant
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int2 sp = clamp(pos + int2(dx, dy), int2(0, 0), int2(w - 1, h - 1));
            
            float2 mv = MotionIn.Load(int3(sp, 0));
            float conf = ConfIn.Load(int3(sp, 0));
            float luma = LumaIn.Load(int3(sp, 0));
            
            // Adjusted sigma for smaller kernel (sigma^2 = 4.0)
            float spatialW = exp(-float(dx * dx + dy * dy) / 4.0);
            float lumaDiff = abs(luma - centerLuma);
            
            // Restore Edge Awareness to prevent Halos around bright lights
            // Use a softer Falloff but DO NOT allow free bleeding across high contrast edges
            float lumaW = exp(-lumaDiff * 4.0); 
            
            float2 mvDiff = mv - centerMV;
            
            // Re-introduce mild motion similarity weight to prevent blending distinct objects
            float mvW = exp(-dot(mvDiff, mvDiff) / 64.0); // Very loose tolerance (8px diff)
            
            float confW = 0.5 + 4.0 * conf; // Give high confidence pixels DOMINANT weight
            
            float weight = spatialW * lumaW * mvW * confW;
            
            sumMV += mv * weight;
            sumConf += conf * weight;
            sumW += weight;
        }
    }
    
    if (sumW > 0.001) {
        float2 smoothedMV = sumMV / sumW;
        float smoothedConf = sumConf / sumW;
        
        // Use smooth motion almost everywhere, only preserve center if it was VERY confident
        // and part of a structural edge
        float preserve = centerConf * smoothstep(0.05, 0.2, edgeStrength);
        preserve = clamp(preserve, 0.0, 0.4); 
        
        MotionOut[id.xy] = lerp(smoothedMV, centerMV, preserve);
        ConfOut[id.xy] = lerp(smoothedConf, centerConf, preserve * 0.5);
    } else {
        MotionOut[id.xy] = centerMV;
        ConfOut[id.xy] = centerConf;
    }
}
