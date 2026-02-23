// ============================================================================
// DEBUG VIEW v2 - Visualization for Motion, Confidence, and Quality Analysis
// ============================================================================

Texture2D<float4> PrevColor  : register(t0);
Texture2D<float4> CurrColor  : register(t1);
Texture2D<float2> Motion     : register(t2);
Texture2D<float>  Confidence : register(t3);
RWTexture2D<float4> OutColor : register(u0);

SamplerState LinearClamp : register(s0);

cbuffer DebugCB : register(b0) {
    int   mode;
    float motionScale;
    float diffScale;
    float pad;
};

// ============================================================================
// HELPERS
// ============================================================================
float3 HSVtoRGB(float h, float s, float v) {
    float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    float3 p = abs(frac(float3(h,h,h) + K.xyz) * 6.0 - K.www);
    return v * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

float3 RGBToYCoCg(float3 c) {
    return float3(
        c.r * 0.25 + c.g * 0.5 + c.b * 0.25,
        c.r * 0.5 - c.b * 0.5,
        -c.r * 0.25 + c.g * 0.5 - c.b * 0.25
    );
}

float3 Turbo(float t) {
    t = saturate(t);
    float r = 0.237 - 2.13*t + 26.92*t*t - 65.5*t*t*t + 63.5*t*t*t*t - 22.36*t*t*t*t*t;
    float g = ((0.572 + 1.524*t - 1.811*t*t) / (1.0 - 0.291*t + 0.1574*t*t)) + 0.003;
    float b = 1.0 / (1.579 - 4.03*t + 12.92*t*t - 31.4*t*t*t + 48.6*t*t*t*t - 23.36*t*t*t*t*t);
    return float3(saturate(r), saturate(g), saturate(b));
}

float LineSDF(float2 p, float2 a, float2 b) {
    float2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// ============================================================================
// MAIN
// ============================================================================
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint outW, outH;
    OutColor.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;

    uint inW, inH;
    CurrColor.GetDimensions(inW, inH);
    uint mvW, mvH;
    Motion.GetDimensions(mvW, mvH);

    float2 outSize = float2(outW, outH);
    float2 inSize  = float2(inW, inH);
    float2 mvSize  = float2(mvW, mvH);

    float2 uv = (float2(id.xy) + 0.5) / outSize;

    float4 curr = CurrColor.SampleLevel(LinearClamp, uv, 0);
    float4 prev = PrevColor.SampleLevel(LinearClamp, uv, 0);
    float2 mv   = Motion.SampleLevel(LinearClamp, uv, 0) * (inSize / mvSize);
    float  conf = Confidence.SampleLevel(LinearClamp, uv, 0);

    float3 finalColor = curr.rgb;

    // MODE 0: Pass-through
    if (mode == 0) {
        finalColor = curr.rgb;
    }
    // MODE 1: Color Flow (HSV)
    else if (mode == 1) {
        float mag = length(mv);
        if (mag < 0.1) {
            finalColor = float3(0, 0, 0);
        } else {
            float angle = atan2(mv.y, mv.x);
            float hue = (angle / 6.283185307) + 0.5;
            float val = saturate(log2(1.0 + mag * 0.2));
            finalColor = HSVtoRGB(hue, 0.9, val);
        }
        finalColor = lerp(curr.rgb * 0.3, finalColor, 0.8);
    }
    // MODE 2: Confidence Heatmap
    else if (mode == 2) {
        float3 heat = Turbo(conf);
        finalColor = heat * (curr.rgb * 0.5 + 0.5);
    }
    // MODE 3: Needle Map
    else if (mode == 3) {
        finalColor = curr.rgb * 0.35;
        float spacing = 32.0;
        float2 gridCenter = (floor(float2(id.xy) / spacing) + 0.5) * spacing;
        float2 gridUV = gridCenter / outSize;
        float2 gridMV = Motion.SampleLevel(LinearClamp, gridUV, 0) * (inSize / mvSize);
        float mag = length(gridMV);

        if (mag > 0.5) {
            float2 arrowEnd = gridCenter + gridMV;
            float d = LineSDF(float2(id.xy), gridCenter, arrowEnd);
            float alpha = 1.0 - smoothstep(0.5, 1.5, d);

            float2 dir = normalize(gridMV);
            float2 headL = arrowEnd - dir * 4.0 + float2(-dir.y, dir.x) * 3.0;
            float2 headR = arrowEnd - dir * 4.0 + float2(dir.y, -dir.x) * 3.0;
            float d2 = min(LineSDF(float2(id.xy), arrowEnd, headL),
                           LineSDF(float2(id.xy), arrowEnd, headR));
            float alpha2 = 1.0 - smoothstep(0.5, 1.5, d2);

            float gridConf = Confidence.SampleLevel(LinearClamp, gridUV, 0);
            float3 arrowColor = float3(0, 1, 0);
            if (gridConf < 0.4 || mag > 100.0) arrowColor = float3(1, 0, 0);
            else if (gridConf < 0.7) arrowColor = float3(1, 1, 0);

            finalColor = lerp(finalColor, arrowColor, max(alpha, alpha2));
        }
    }
    // MODE 4: Residual Error (Warp Check)
    else if (mode == 4) {
        float2 warpUV = uv - (mv / inSize);
        float4 warpedPrev = PrevColor.SampleLevel(LinearClamp, warpUV, 0);
        float3 diff = abs(curr.rgb - warpedPrev.rgb);
        float err = dot(diff, float3(1, 1, 1));
        finalColor = float3(err, err * 0.1, 0) * 4.0;
    }
    // MODE 5: Split Screen
    else if (mode == 5) {
        if (uv.x < 0.5) {
            finalColor = curr.rgb;
        } else {
            float2 warpUV = uv - (mv / inSize);
            finalColor = PrevColor.SampleLevel(LinearClamp, warpUV, 0).rgb;
            if (abs(uv.x - 0.5) < 0.002) finalColor = float3(1, 1, 0);
        }
    }
    // MODE 6: Occlusion Mask
    else if (mode == 6) {
        float2 warpUV = uv - (mv / inSize);
        float4 warpedPrev = PrevColor.SampleLevel(LinearClamp, warpUV, 0);
        float3 diff = abs(curr.rgb - warpedPrev.rgb);
        float occ = exp(-length(diff) * 6.0);
        finalColor = float3(occ, occ, occ);
    }
    // MODE 7: Ghost Detection
    else if (mode == 7) {
        float3 yPrev = RGBToYCoCg(prev.rgb);
        float3 yCurr = RGBToYCoCg(curr.rgb);
        float patchDiff = length(yPrev - yCurr);
        float ghostFactor = smoothstep(0.05 * diffScale, 0.25 * diffScale, patchDiff);
        finalColor = lerp(float3(0, 0, 0.5), float3(1, 0, 0), ghostFactor);
        finalColor += curr.rgb * 0.2;
    }
    // MODE 8: Structure Gradient
    else if (mode == 8) {
        float v00 = RGBToYCoCg(CurrColor.SampleLevel(LinearClamp, uv, 0, int2(-1,-1)).rgb).x;
        float v10 = RGBToYCoCg(CurrColor.SampleLevel(LinearClamp, uv, 0, int2( 0,-1)).rgb).x;
        float v20 = RGBToYCoCg(CurrColor.SampleLevel(LinearClamp, uv, 0, int2( 1,-1)).rgb).x;
        float v01 = RGBToYCoCg(CurrColor.SampleLevel(LinearClamp, uv, 0, int2(-1, 0)).rgb).x;
        float v21 = RGBToYCoCg(CurrColor.SampleLevel(LinearClamp, uv, 0, int2( 1, 0)).rgb).x;
        float v02 = RGBToYCoCg(CurrColor.SampleLevel(LinearClamp, uv, 0, int2(-1, 1)).rgb).x;
        float v12 = RGBToYCoCg(CurrColor.SampleLevel(LinearClamp, uv, 0, int2( 0, 1)).rgb).x;
        float v22 = RGBToYCoCg(CurrColor.SampleLevel(LinearClamp, uv, 0, int2( 1, 1)).rgb).x;

        float gx = -v00 + v20 - 2.0*v01 + 2.0*v21 - v02 + v22;
        float gy = -v00 - 2.0*v10 - v20 + v02 + 2.0*v12 + v22;
        float mag = length(float2(gx, gy)) * 4.0;
        finalColor = (mag > 0.5) ? float3(0, mag, 0) : float3(mag, mag, mag);
    }

    OutColor[id.xy] = float4(finalColor, 1.0);
}
