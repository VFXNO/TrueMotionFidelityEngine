
import os

file_path = r'c:/Users/vfxno/Downloads/2/Frame_interpolation (gaming)/src/shaders/MotionTemporal.hlsl'

# 1. Verify existence
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

# 2. Check current content snippet
with open(file_path, 'r') as f:
    content = f.read()
    print(f"Current file start: {content[:50]}")

# 3. New Content (TAA v2)
new_content = r'''// ============================================================================
// MOTION TAA v2 - Robust Vector Stabilization
// Implements Neighborhood Clamping + Motion Reprojection
// ============================================================================

Texture2D<float2> MotionCurr : register(t0);    // Current Raw/Smoothed Motion (Input)
Texture2D<float> ConfCurr : register(t1);
Texture2D<float2> MotionHistory : register(t2); // Previous Frame's Result (History)
Texture2D<float> ConfHistory : register(t3);
Texture2D<float> LumaPrev : register(t4);
Texture2D<float> LumaCurr : register(t5);

RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> ConfOut : register(u1);

SamplerState LinearClamp : register(s0);

cbuffer TemporalCB : register(b0) {
    float historyWeight;
    float confInfluence;
    int resetHistory;
    float pad;
};

// ============================================================================
// TAA LOGIC
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    MotionCurr.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height) return;
    
    float2 texelSize = 1.0 / float2(width, height);
    float2 uv = (float2(id.xy) + 0.5) * texelSize;
    
    // ------------------------------------------------------------------------
    // 1. Load Current Data & Build Neighborhood
    // ------------------------------------------------------------------------
    float2 currMotion = MotionCurr.Load(int3(id.xy, 0));
    float currConf = ConfCurr.Load(int3(id.xy, 0));
    
    // Find Min/Max in 3x3 neighborhood for clamping (Bounding Box)
    float2 minMotion = currMotion;
    float2 maxMotion = currMotion;
    
    [unroll]
    for(int y = -1; y <= 1; ++y) {
        [unroll]
        for(int x = -1; x <= 1; ++x) {
            // Sample neighbors (using Point load for exact values)
            int2 nPos = id.xy + int2(x, y);
            nPos = clamp(nPos, int2(0, 0), int2(width-1, height-1));
            float2 nMotion = MotionCurr.Load(int3(nPos, 0));
            
            minMotion = min(minMotion, nMotion);
            maxMotion = max(maxMotion, nMotion);
        }
    }
    
    // ------------------------------------------------------------------------
    // 2. Temporal Reprojection
    // ------------------------------------------------------------------------
    // Use current motion to find where this surface was in the previous frame
    // Motion V points from Prev -> Curr. So PrevPos = CurrPos - V
    float2 historyUV = uv - (currMotion * texelSize);
    
    // Valid history check
    bool validHistory = !resetHistory;
    if (historyUV.x < 0 || historyUV.y < 0 || historyUV.x > 1 || historyUV.y > 1) {
        validHistory = false;
    }
    
    if (!validHistory) {
        MotionOut[id.xy] = currMotion;
        ConfOut[id.xy] = currConf;
        return;
    }
    
    // ------------------------------------------------------------------------
    // 3. Sample History & Clamping
    // ------------------------------------------------------------------------
    float2 historyMotion = MotionHistory.SampleLevel(LinearClamp, historyUV, 0);
    float historyConf = ConfHistory.SampleLevel(LinearClamp, historyUV, 0);
    
    // CLAMP: The history vector must be somewhat similar to local neighbors
    // This removes "ghost" vectors from disoccluded objects
    float2 clampedHistory = clamp(historyMotion, minMotion, maxMotion);
    
    // ------------------------------------------------------------------------
    // 4. Adaptive Blending
    // ------------------------------------------------------------------------
    // Base blend: trusting history heavily (0.94) for stability
    float alpha = 0.94;
    
    // Calculate deviation
    float2 diff = clampedHistory - currMotion;
    float diffLen = length(diff);
    
    // If the history (even after clamping) is very different from current,
    // it implies fast acceleration or error. Reduce history trust.
    if (diffLen > 2.0) {
        alpha = 0.8;
    }
    if (diffLen > 10.0) {
        alpha = 0.5;
    }
    
    // Integration of Confidence
    // If current frame is high confidence, we can accept it more readily
    if (currConf > 0.9) {
        alpha *= 0.9; // Allow slightly faster convergence to high-conf input
    }
    
    // If history was low confidence, don't keep it alive
    if (historyConf < 0.2) {
        alpha = 0.0; // Reset
    }
    
    // ------------------------------------------------------------------------
    // 5. Output
    // ------------------------------------------------------------------------
    float2 finalMotion = lerp(currMotion, clampedHistory, alpha);
    float finalConf = lerp(currConf, historyConf, alpha);
    
    // Anti-Jitter for static scenes
    // If result is tiny, kill it to prevent sub-pixel crawling
    if (length(finalMotion) < 0.1) {
        finalMotion = float2(0, 0);
    }
    
    MotionOut[id.xy] = finalMotion;
    ConfOut[id.xy] = finalConf;
}
'''

with open(file_path, 'w') as f:
    f.write(new_content)

print("Successfully overwrote MotionTemporal.hlsl with TAA v2 logic.")
