cbuffer Params : register(b0)
{
    float width;
    float height;
    float alpha;
    float motionScale;
    
    int debugMode;
    float debugScale;
    float confidenceThreshold;
    int motionModel;
};

Texture2D<float4> TexPrev : register(t0);
Texture2D<float4> TexCurr : register(t1);
Texture2D<float2> TexFlowInput : register(t2);

RWTexture2D<float2> OutFlow : register(u0);
RWTexture2D<float4> OutFrame : register(u0); // Note: register u0 reused in different passes

SamplerState LinearClamp : register(s0);

// Helper to get luminance
float Luma(float3 color) {
    return dot(color, float3(0.299, 0.587, 0.114));
}

// Pass 1: Simple Block Matching Optical Flow
// This is a simplified search. For production quality, use hierarchical/pyramid search.
[numthreads(8, 8, 1)]
void CSOpticalFlow(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;

    int2 pos = int2(id.x, id.y);
    float4 centerPixel = TexPrev.Load(int3(pos, 0));
    float centerLuma = Luma(centerPixel.rgb);

    // Search window
    int radius = 4; // Search radius
    float minDiff = 10000.0;
    int2 bestVec = int2(0, 0);

    // Naive block matching - very expensive if radius is large
    // Optimized for this example: small radius, search for brightness constancy
    for (int y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            int2 searchPos = pos + int2(x, y);
            if (searchPos.x < 0 || searchPos.x >= width || searchPos.y < 0 || searchPos.y >= height) continue;

            float4 targetPixel = TexCurr.Load(int3(searchPos, 0));
            float diff = abs(Luma(targetPixel.rgb) - centerLuma); 
            // Add gradient term or structure compare for better quality
            
            if (diff < minDiff)
            {
                minDiff = diff;
                bestVec = int2(x, y);
            }
        }
    }

    OutFlow[pos] = float2(bestVec.x, bestVec.y);
}

// Pass 2: Frame Interpolation using Motion Vectors
[numthreads(8, 8, 1)]
void CSInterpolate(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;

    int2 pos = int2(id.x, id.y);

    // Fetch motion vector for this pixel
    // Ideally we sample flow at the interpolated position, but forward/backward mapping is complex.
    // Using simple backward mapping from grid.
    float2 flow = TexFlowInput.Load(int3(pos, 0));
    
    // Bidirectional flow: P -> C is flow.
    // To get frame at alpha (0..1), we sample:
    // Prev at (pos - flow * alpha) roughly? 
    // Actually, if flow is P->C, then P(p) moves to C(p+flow).
    // Interpolated pixel at p' approx p + flow*alpha.
    // Here we use simple blending with displacement.
    
    float2 uv = (float2(pos) + 0.5) / float2(width, height);
    float2 texelSize = 1.0 / float2(width, height);

    // Scaled flow
    float2 flowVec = flow * texelSize; 

    float2 uvPrev = uv - flowVec * alpha; // Look back along flow?
    float2 uvCurr = uv + flowVec * (1.0 - alpha); // Look forward?

    float4 c0 = TexPrev.SampleLevel(LinearClamp, uvPrev, 0);
    float4 c1 = TexCurr.SampleLevel(LinearClamp, uvCurr, 0);

    float4 result = lerp(c0, c1, alpha);
    
    OutFrame[pos] = float4(result.rgb, 1.0); 
}

// Pass 3: Debug View
[numthreads(8, 8, 1)]
void CSDebug(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    int2 pos = int2(id.x, id.y);

    if (debugMode == 1) // Motion Flow
    {
        float2 flow = TexFlowInput.Load(int3(pos, 0));
        // Visualize flow as color
        // R = x, G = y
        float2 vis = (flow * debugScale) * 0.5 + 0.5;
        OutFrame[pos] = float4(vis.x, vis.y, 0.5, 1.0);
    }
    else if (debugMode == 2) // Confidence/Heatmap (Difference)
    {
        float4 c0 = TexPrev.Load(int3(pos, 0));
        float4 c1 = TexCurr.Load(int3(pos, 0));
        float diff = length(c0.rgb - c1.rgb);
        OutFrame[pos] = float4(diff * debugScale, 0, 0, 1);
    }
    else
    {
        OutFrame[pos] = TexCurr.Load(int3(pos, 0));
    }
}
