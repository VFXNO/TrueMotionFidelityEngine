// ============================================================================
// PROFESSIONAL DEBUG VIEW v3
// Advanced visualization for Motion Vectors and Confidence
// ============================================================================

Texture2D<float4> PrevColor : register(t0);
Texture2D<float4> CurrColor : register(t1);
Texture2D<float2> Motion : register(t2);
Texture2D<float> Confidence : register(t3);
RWTexture2D<float4> OutColor : register(u0);

SamplerState LinearClamp : register(s0);

cbuffer DebugCB : register(b0) {
    int mode;           // 0=Clear, 1=Flow, 2=Confidence, 3=Needles, 4=WarpCheck, 5=Split, 6=Heat
    float motionScale;  // Scale for vis
    float diffScale;    // Contrast
    float pad;
};

// ============================================================================
// UTILS
// ============================================================================
float3 HSVtoRGB(float h, float s, float v) {
    float3 c = float3(h, s, v);
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Turbo Colormap for Heatmaps (Blue -> Green -> Red)
float3 Turbo(float t) {
    t = saturate(t);
    return float3(
        0.237 - 2.13*t + 26.92*t*t - 65.5*t*t*t + 63.5*t*t*t*t - 22.36*t*t*t*t*t,
        ((0.572 + 1.524*t - 1.811*t*t) / (1.0 - 0.291*t + 0.1574*t*t)) + 0.003,
        1.0 / (1.579 - 4.03*t + 12.92*t*t - 31.4*t*t*t + 48.6*t*t*t*t - 23.36*t*t*t*t*t)
    );
}

// Draw line segment (SDF)
float Line(float2 p, float2 a, float2 b) {
    float2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// ============================================================================
// MAIN SHADER
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutColor.GetDimensions(outW, outH);
    
    if (id.x >= outW || id.y >= outH) return;
    
    uint inW, inH;
    CurrColor.GetDimensions(inW, inH);
    uint mvW, mvH;
    Motion.GetDimensions(mvW, mvH); // Usually quarter res
    
    float2 outSize = float2(outW, outH);
    float2 inSize = float2(inW, inH);
    float2 mvSize = float2(mvW, mvH);
    
    float2 uv = (float2(id.xy) + 0.5) / outSize;
    
    // Sample Source
    float4 curr = CurrColor.SampleLevel(LinearClamp, uv, 0);
    float4 prev = PrevColor.SampleLevel(LinearClamp, uv, 0);
    
    // Sample Motion (Rescale to input resolution pixels)
    float2 mv = Motion.SampleLevel(LinearClamp, uv, 0);
    mv *= (inSize / mvSize); // Convert from MV-texture-space to Screen-Pixels
    
    float conf = Confidence.SampleLevel(LinearClamp, uv, 0);
    
    float3 finalColor = curr.rgb;
    
    // ========================================================================
    // MODE 0: ORIGINAL (Pass-through)
    // ========================================================================
    if (mode == 0) {
        finalColor = curr.rgb;
    }
    
    // ========================================================================
    // MODE 1: COLOR FLOW (Professional HSV)
    // Hue = Direction, Saturation = 1.0, Value = Speed
    // ========================================================================
    else if (mode == 1) {
        float mag = length(mv);
        
        if (mag < 0.1) {
            finalColor = float3(0, 0, 0); // Black for static
        } else {
            float angle = atan2(mv.y, mv.x);
            float hue = (angle / 6.283185307) + 0.5;
            float val = saturate(log2(1.0 + mag * 0.2)); 
            finalColor = HSVtoRGB(hue, 0.9, val);
        }
        finalColor = lerp(curr.rgb * 0.3, finalColor, 0.8);
    }
    
    // ========================================================================
    // MODE 2: CONFIDENCE HEATMAP Error Check
    // Red = Bad (0.0), Yellow = Medium, Green = Good (1.0)
    // ========================================================================
    else if (mode == 2) {
        float3 heat = lerp(float3(1,0,0), float3(1,1,0), smoothstep(0.0, 0.5, conf));
        heat = lerp(heat, float3(0,1,0), smoothstep(0.5, 1.0, conf));
        finalColor = heat;
        // Show structure
        finalColor *= (curr.rgb * 0.5 + 0.5);
    }
    
    // ========================================================================
    // MODE 3: NEEDLE MAP (With Outlier Detection)
    // Green = Good flow. Red = Outlier/Suspect.
    // ========================================================================
    else if (mode == 3) {
        finalColor = curr.rgb * 0.4; // Dim background
        
        float spacing = 32.0;
        float2 gridCenter = (floor(float2(id.xy) / spacing) + 0.5) * spacing;
        
        float2 gridUV = gridCenter / outSize;
        float2 gridMV = Motion.SampleLevel(LinearClamp, gridUV, 0);
        gridMV *= (inSize / mvSize); 
        
        float mag = length(gridMV);
        
        if (mag > 0.5) {
            float2 arrowStart = gridCenter;
            float2 arrowEnd = gridCenter + gridMV; 
            
            float d = Line(float2(id.xy), arrowStart, arrowEnd);
            float alpha = 1.0 - smoothstep(0.5, 1.5, d);
            
            float2 dir = normalize(gridMV);
            float2 headL = arrowEnd - dir * 4.0 + float2(-dir.y, dir.x) * 3.0;
            float2 headR = arrowEnd - dir * 4.0 + float2(dir.y, -dir.x) * 3.0;
            
            float d2 = Line(float2(id.xy), arrowEnd, headL);
            d2 = min(d2, Line(float2(id.xy), arrowEnd, headR));
            float alpha2 = 1.0 - smoothstep(0.5, 1.5, d2);
            
            // Color logic: if motion is large but confidence is low -> RED
            float3 arrowColor = float3(0,1,0); 
            float gridConf = Confidence.SampleLevel(LinearClamp, gridUV, 0);
            
            if (gridConf < 0.4 || mag > 100.0) arrowColor = float3(1,0,0);
            else if (gridConf < 0.7) arrowColor = float3(1,1,0);
            
            finalColor = lerp(finalColor, arrowColor, max(alpha, alpha2));
        }
    }
    
    // ========================================================================
    // MODE 4: RESIDUAL ERROR (Warp Check)
    // Black = Perfect. Red = Error.
    // ========================================================================
    else if (mode == 4) {
        float2 warpUV = uv - (mv / inSize); 
        float4 warpedPrev = PrevColor.SampleLevel(LinearClamp, warpUV, 0);
        
        float3 diff = abs(curr.rgb - warpedPrev.rgb);
        float err = dot(diff, float3(1,1,1)); 
        
        finalColor = float3(err, err*0.1, 0) * 4.0; 
    }
    
    // ========================================================================
    // MODE 5: SPLIT SCREEN (Source Left | Warped Right)
    // Useful to see if warping aligns with reality
    // ========================================================================
    else if (mode == 5) {
        if (uv.x < 0.5) {
            finalColor = curr.rgb; // Left: Reality
        } else {
            // Right: Prediction (Warped Prev)
            float2 warpUV = uv - (mv / inSize); 
            // Shift UV back to 0-1 range for the split
            // actually just warp the right side pixels
            finalColor = PrevColor.SampleLevel(LinearClamp, warpUV, 0).rgb;
            
            // Draw border
            if (abs(uv.x - 0.5) < 0.002) finalColor = float3(1,1,0);
        }
    }
    
    // ========================================================================
    // MODE 6: OCCLUSION MASK
    // White = Visible. Black = Occluded.
    // ========================================================================
    else if (mode >= 6) {
        float2 warpUV = uv - (mv / inSize); 
        float4 warpedPrev = PrevColor.SampleLevel(LinearClamp, warpUV, 0);
        float3 diff = abs(curr.rgb - warpedPrev.rgb);
        float occlusion = (length(diff) > 0.25) ? 0.0 : 1.0;
        
        finalColor = float3(occlusion, occlusion, occlusion);
    }

    OutColor[id.xy] = float4(finalColor, 1.0);
}
