// ============================================================================
// MOTION SMOOTHING v2 - Joint Bilateral Filter
//
// Key improvements:
//   1. Proper joint bilateral filter with three edge-stopping functions:
//      spatial distance, luma similarity, and motion coherence
//   2. Confidence-weighted accumulation prevents bad vectors from spreading
//   3. Adaptive kernel size: large in smooth regions, small near edges
//   4. Preserves sharp motion boundaries (no halo bleeding)
// ============================================================================

Texture2D<float2> MotionIn : register(t0);
Texture2D<float>  ConfIn   : register(t1);
Texture2D<float>  LumaIn   : register(t2);
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float>  ConfOut   : register(u1);

cbuffer SmoothCB : register(b0) {
    float edgeScale;
    float confPower;
    float2 pad;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    MotionIn.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;

    int2 pos = int2(id.xy);
    int2 maxPos = int2(int(w) - 1, int(h) - 1);

    float2 centerMV   = MotionIn.Load(int3(pos, 0));
    float  centerConf = ConfIn.Load(int3(pos, 0));
    float  centerLuma = LumaIn.Load(int3(pos, 0));

    // Compute local edge strength for adaptive kernel
    float lL = LumaIn.Load(int3(clamp(pos + int2(-1, 0), int2(0,0), maxPos), 0));
    float lR = LumaIn.Load(int3(clamp(pos + int2( 1, 0), int2(0,0), maxPos), 0));
    float lU = LumaIn.Load(int3(clamp(pos + int2( 0,-1), int2(0,0), maxPos), 0));
    float lD = LumaIn.Load(int3(clamp(pos + int2( 0, 1), int2(0,0), maxPos), 0));
    float edgeStr = abs(lR - lL) + abs(lD - lU);

    // Sigma parameters for the bilateral filter
    float sigmaSpatial = lerp(2.5, 1.0, saturate(edgeStr * edgeScale));
    float sigmaLuma    = lerp(0.08, 0.03, saturate(edgeStr * edgeScale));
    float sigmaMotion  = lerp(4.0, 1.5, saturate(edgeStr * edgeScale));

    float invSigmaSpatial2 = 1.0 / (2.0 * sigmaSpatial * sigmaSpatial);
    float invSigmaLuma2    = 1.0 / (2.0 * sigmaLuma * sigmaLuma);
    float invSigmaMotion2  = 1.0 / (2.0 * sigmaMotion * sigmaMotion);

    // Adaptive kernel radius: smaller near edges
    int kernelR = (edgeStr > 0.15) ? 1 : 2;

    float2 sumMV   = float2(0, 0);
    float  sumConf = 0;
    float  sumW    = 0;

    [loop] for (int dy = -kernelR; dy <= kernelR; ++dy) {
        [loop] for (int dx = -kernelR; dx <= kernelR; ++dx) {
            int2 sp = clamp(pos + int2(dx, dy), int2(0, 0), maxPos);

            float2 mv   = MotionIn.Load(int3(sp, 0));
            float  conf = ConfIn.Load(int3(sp, 0));
            float  luma = LumaIn.Load(int3(sp, 0));

            // 1. Spatial distance weight (Gaussian)
            float dist2 = float(dx * dx + dy * dy);
            float wSpatial = exp(-dist2 * invSigmaSpatial2);

            // 2. Luma similarity weight (edge-stopping)
            float lumaDiff = abs(luma - centerLuma);
            float wLuma = exp(-lumaDiff * lumaDiff * invSigmaLuma2);

            // 3. Motion coherence weight (prevents blending across motion boundaries)
            float2 mvDiff = mv - centerMV;
            float mvDist2 = dot(mvDiff, mvDiff);
            float wMotion = exp(-mvDist2 * invSigmaMotion2);

            // 4. Confidence boost (trust high-confidence neighbors more)
            float wConf = 0.2 + 0.8 * pow(saturate(conf), confPower);

            float weight = wSpatial * wLuma * wMotion * wConf;

            sumMV   += mv * weight;
            sumConf += conf * weight;
            sumW    += weight;
        }
    }

    if (sumW > 1e-4) {
        float2 smoothMV   = sumMV / sumW;
        float  smoothConf = sumConf / sumW;

        // Preserve center motion on strong edges with high confidence
        float preserve = centerConf * smoothstep(0.08, 0.25, edgeStr);
        preserve = clamp(preserve, 0.0, 0.5);

        MotionOut[id.xy] = lerp(smoothMV, centerMV, preserve);
        ConfOut[id.xy]   = lerp(smoothConf, centerConf, preserve * 0.5);
    } else {
        MotionOut[id.xy] = centerMV;
        ConfOut[id.xy]   = centerConf;
    }
}
