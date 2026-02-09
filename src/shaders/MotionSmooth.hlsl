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
    
    // Increased kernel size 3x3 -> 5x5 for better stability
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int2 sp = clamp(pos + int2(dx, dy), int2(0, 0), int2(w - 1, h - 1));
            
            float2 mv = MotionIn.Load(int3(sp, 0));
            float conf = ConfIn.Load(int3(sp, 0));
            float luma = LumaIn.Load(int3(sp, 0));
            
            float spatialW = exp(-float(dx * dx + dy * dy) / 4.0); // Broader spatial falloff
            float lumaDiff = abs(luma - centerLuma);
            float lumaW = exp(-lumaDiff / max(0.05, edgeScale * 0.2)); // Relaxed luma constraint
            
            float2 mvDiff = mv - centerMV;
            // Key fix: If center is outlier, we don't want to exclude neighbors that differ from it.
            // Instead of penalizing difference from center, we penalize difference from 'neighbors average' (too loop heavy)
            // Or just relax this weight significantly so we average distinct vectors together (smoothing major discontinuities)
            float mvW = exp(-dot(mvDiff, mvDiff) / 16.0); 
            
            float confW = 0.5 + 2.0 * conf; // Give high confidence pixels much more weight
            
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
        preserve = clamp(preserve, 0.0, 0.5); // Max 50% blend with original - prefer smoothed
        
        MotionOut[id.xy] = lerp(smoothedMV, centerMV, preserve);
        ConfOut[id.xy] = lerp(smoothedConf, centerConf, preserve * 0.5);
    } else {
        MotionOut[id.xy] = centerMV;
        ConfOut[id.xy] = centerConf;
    }
}
