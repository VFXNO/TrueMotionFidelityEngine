#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <array>
#include <random>
#include <ctime>
#include <chrono>
#include <shlwapi.h>
#include <shlobj.h>
#include <dwmapi.h>
#include <fstream>
#include <mutex>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "dwmapi.lib")

#include <mutex>

struct Config {
    std::string inputFolder;
    std::string outputFolder;
    std::string rifePath;
    int numFrames = 1000;
    int epochs = 500;
    float learningRate = 0.005f;
    int batchSize = 32;
    int stride = 1; // Added stride parameter
    bool cleanStatic = true;
    bool useRIFE = false;
};

struct Weights {
    // === IFNet-Lite: 12->8->16 MLP ===
    // Hidden layer weights: 8 units, each with 4 shared weights
    float mlpW_h0[4], mlpW_h1[4], mlpW_h2[4], mlpW_h3[4];
    float mlpW_h4[4], mlpW_h5[4], mlpW_h6[4], mlpW_h7[4];
    // Output weights: 4 shared vectors
    float mlpW_out0[4], mlpW_out1[4], mlpW_out2[4], mlpW_out3[4];
    // Hidden biases (8 values in 2 float4)
    float mlpBias_h0[4];  // h0,h1,h2,h3
    float mlpBias_h1[4];  // h4,h5,h6,h7
    // Output biases (16 values in 4 float4)
    float mlpBias_out0[4]; // attention gates 1-4
    float mlpBias_out1[4]; // attention gates 5-8
    float mlpBias_out2[4]; // attention gates 9-12
    float mlpBias_out3[4]; // residual_dx, residual_dy, occlusion, quality
    // Base attention priors
    float baseW1[4], baseW2[4], baseW3[4];
    // === FusionNet-Lite: 12->6->4 Synthesis MLP ===
    float synthW_h0[4], synthW_h1[4], synthW_h2[4];
    float synthW_h3[4], synthW_h4[4], synthW_h5[4];
    float synthW_out0[4], synthW_out1[4];
    float synthBias_h0[4], synthBias_h1[4];
    float synthBias_out[4];
    // Flag + padding
    float useCustomWeights;
    float _pad[3];

    static constexpr int NUM_PARAMS = 128;
    float* params() { return &mlpW_h0[0]; }
    const float* params() const { return &mlpW_h0[0]; }
};

std::atomic<bool> g_training(false);
std::atomic<int> g_progress(0);
std::vector<std::string> g_log;
std::mutex g_log_mutex;
std::atomic<bool> g_log_updated(false);

Config g_cfg;
Weights g_weights;

const COLORREF BG_COLOR = RGB(243, 243, 243);
const COLORREF CARD_COLOR = RGB(255, 255, 255);
const COLORREF ACCENT_COLOR = RGB(0, 120, 212);
const COLORREF ACCENT_LIGHT = RGB(153, 211, 253);
const COLORREF TEXT_COLOR = RGB(32, 32, 32);
const COLORREF TEXT_SECONDARY = RGB(136, 136, 136);
const COLORREF BORDER_COLOR = RGB(229, 229, 229);
const COLORREF INPUT_BG = RGB(251, 251, 251);
const COLORREF PROGRESS_BG = RGB(229, 229, 229);

bool fileExists(const std::string& path) {
    return GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES;
}

void createFolder(const char* path) { CreateDirectoryA(path, nullptr); }

std::vector<std::string> getFiles(const std::string& folder, const std::vector<std::string>& exts) {
    std::vector<std::string> files;
    std::string searchPath = folder + "\\*.*";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                std::string name = fd.cFileName;
                size_t dot = name.rfind('.');
                if (dot != std::string::npos) {
                    std::string ext = name.substr(dot + 1);
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    for (const auto& e : exts) {
                        if (ext == e) {
                            files.push_back(folder + "\\" + name);
                            break;
                        }
                    }
                }
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
    }
    std::sort(files.begin(), files.end());
    return files;
}

void log(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log.push_back(msg);
    if (g_log.size() > 500) g_log.erase(g_log.begin());
    g_log_updated = true;
}

// Forward pass matching the shader's CnnAttention12 with custom weights
void forward(const Weights& w, const float* feat, float* outW) {
    // Normalize inputs (shader: total = dot(e1,1)+dot(e2,1)+dot(e3,1)+eps; x = max(e/total, 0))
    float total = 1e-6f;
    for (int j = 0; j < 12; j++) total += std::max(0.0f, feat[j]);
    float nx[12];
    for (int j = 0; j < 12; j++) nx[j] = std::max(0.0f, feat[j] / total);
    const float* x1 = nx;
    const float* x2 = nx + 4;
    const float* x3 = nx + 8;

    const float* wh[8] = { w.mlpW_h0, w.mlpW_h1, w.mlpW_h2, w.mlpW_h3,
                            w.mlpW_h4, w.mlpW_h5, w.mlpW_h6, w.mlpW_h7 };
    float hBias[8] = { w.mlpBias_h0[0], w.mlpBias_h0[1], w.mlpBias_h0[2], w.mlpBias_h0[3],
                       w.mlpBias_h1[0], w.mlpBias_h1[1], w.mlpBias_h1[2], w.mlpBias_h1[3] };

    // Hidden layer: 12 -> 8 (ReLU) with shader's weight-sharing permutations
    float h[8];
    for (int i = 0; i < 8; i++) {
        float sum = 0;
        // dot(x1, wh[i].xyzw)
        for (int j = 0; j < 4; j++) sum += x1[j] * wh[i][j];
        // dot(x2, wh[i].zwxy * {-1,-1,1,1})
        sum -= x2[0] * wh[i][2]; sum -= x2[1] * wh[i][3];
        sum += x2[2] * wh[i][0]; sum += x2[3] * wh[i][1];
        // dot(x3, wh[i].wxyz)
        sum += x3[0] * wh[i][3]; sum += x3[1] * wh[i][0];
        sum += x3[2] * wh[i][1]; sum += x3[3] * wh[i][2];
        h[i] = std::max(0.0f, sum + hBias[i]);
    }

    // Output layer: 8 -> 16 (saturate) using 4 output weight vectors
    // Group 1 (out0-3): h0..h3 via mlpW_out0..out2 + mlpW_out3
    float z[12];
    for (int j = 0; j < 4; j++) {
        z[j] = std::clamp(w.mlpW_out0[j]*h[0] + w.mlpW_out1[j]*h[1] + w.mlpW_out2[j]*h[2] + w.mlpW_out3[j]*h[3] + w.mlpBias_out0[j], 0.0f, 1.0f);
    }
    // Group 2 (out4-7): h4..h7
    for (int j = 0; j < 4; j++) {
        z[4+j] = std::clamp(w.mlpW_out0[j]*h[4] + w.mlpW_out1[j]*h[5] + w.mlpW_out2[j]*h[6] + w.mlpW_out3[j]*h[7] + w.mlpBias_out1[j], 0.0f, 1.0f);
    }
    // Group 3 (out8-11): mixed pairs (h0+h1), (h2+h3), (h4+h5), (h6+h7)
    float z3b[4] = { w.mlpBias_out2[0], w.mlpBias_out2[1], w.mlpBias_out2[2], w.mlpBias_out2[3] };
    for (int j = 0; j < 4; j++)
        z[8+j] = std::clamp(w.mlpW_out0[j]*(h[0]+h[1]) + w.mlpW_out1[j]*(h[2]+h[3]) + w.mlpW_out2[j]*(h[4]+h[5]) + w.mlpW_out3[j]*(h[6]+h[7]) + z3b[j], 0.0f, 1.0f);

    // Gate x prior, normalize
    float sumW = 1e-6f;
    for (int j = 0; j < 4; j++) {
        outW[j]   = z[j]   * (w.baseW1[j] + 0.35f * x1[j]); sumW += outW[j];
        outW[4+j] = z[4+j] * (w.baseW2[j] + 0.35f * x2[j]); sumW += outW[4+j];
        outW[8+j] = z[8+j] * (w.baseW3[j] + 0.35f * x3[j]); sumW += outW[8+j];
    }
    for (int j = 0; j < 12; j++) outW[j] /= sumW;
}

// IFNet-Lite extra outputs: motion residual, occlusion, quality
// Uses antisymmetric hidden unit difference combinations
void forwardExtra(const Weights& w, const float* feat, float* motionRes, float* occlusion, float* quality) {
    // Run same hidden layer as forward()
    float total = 1e-6f;
    for (int j = 0; j < 12; j++) total += std::max(0.0f, feat[j]);
    float nx[12];
    for (int j = 0; j < 12; j++) nx[j] = std::max(0.0f, feat[j] / total);
    const float* x1 = nx;
    const float* x2 = nx + 4;
    const float* x3 = nx + 8;

    const float* wh[8] = { w.mlpW_h0, w.mlpW_h1, w.mlpW_h2, w.mlpW_h3,
                            w.mlpW_h4, w.mlpW_h5, w.mlpW_h6, w.mlpW_h7 };
    float hBias[8] = { w.mlpBias_h0[0], w.mlpBias_h0[1], w.mlpBias_h0[2], w.mlpBias_h0[3],
                       w.mlpBias_h1[0], w.mlpBias_h1[1], w.mlpBias_h1[2], w.mlpBias_h1[3] };

    float h[8];
    for (int i = 0; i < 8; i++) {
        float sum = 0;
        for (int j = 0; j < 4; j++) sum += x1[j] * wh[i][j];
        sum -= x2[0] * wh[i][2]; sum -= x2[1] * wh[i][3];
        sum += x2[2] * wh[i][0]; sum += x2[3] * wh[i][1];
        sum += x3[0] * wh[i][3]; sum += x3[1] * wh[i][0];
        sum += x3[2] * wh[i][1]; sum += x3[3] * wh[i][2];
        h[i] = std::max(0.0f, sum + hBias[i]);
    }

    // Extra output group: antisymmetric differences (h0-h4, h1-h5, h2-h6, h3-h7)
    float extra[4];
    for (int j = 0; j < 4; j++)
        extra[j] = w.mlpW_out0[j]*(h[0]-h[4]) + w.mlpW_out1[j]*(h[1]-h[5]) + w.mlpW_out2[j]*(h[2]-h[6]) + w.mlpW_out3[j]*(h[3]-h[7]) + w.mlpBias_out3[j];

    // motion residual: tanh * 0.5
    motionRes[0] = std::tanh(extra[0]) * 0.5f;
    motionRes[1] = std::tanh(extra[1]) * 0.5f;
    // occlusion: sigmoid
    *occlusion = 1.0f / (1.0f + std::exp(-extra[2]));
    // quality: sigmoid
    *quality = 1.0f / (1.0f + std::exp(-extra[3]));
}

// FusionNet-Lite: 12->6->4 synthesis forward pass
void synthForward(const Weights& w, const float* diff, float* out4) {
    float total = 1e-6f;
    for (int j = 0; j < 12; j++) total += std::max(0.0f, diff[j]);
    float nx[12];
    for (int j = 0; j < 12; j++) nx[j] = std::max(0.0f, diff[j] / total);
    const float* d1 = nx;
    const float* d2 = nx + 4;
    const float* d3 = nx + 8;

    const float* sh[6] = { w.synthW_h0, w.synthW_h1, w.synthW_h2, w.synthW_h3, w.synthW_h4, w.synthW_h5 };
    float sBias[6] = { w.synthBias_h0[0], w.synthBias_h0[1], w.synthBias_h0[2], w.synthBias_h0[3],
                       w.synthBias_h1[0], w.synthBias_h1[1] };

    float h[6];
    for (int i = 0; i < 6; i++) {
        float sum = 0;
        for (int j = 0; j < 4; j++) sum += d1[j] * sh[i][j];
        sum -= d2[0] * sh[i][2]; sum -= d2[1] * sh[i][3];
        sum += d2[2] * sh[i][0]; sum += d2[3] * sh[i][1];
        sum += d3[0] * sh[i][3]; sum += d3[1] * sh[i][0];
        sum += d3[2] * sh[i][1]; sum += d3[3] * sh[i][2];
        h[i] = std::max(0.0f, sum + sBias[i]);
    }

    // Output: 6->4 (sigmoid)
    for (int j = 0; j < 4; j++) {
        float val = w.synthW_out0[j]*h[0] + w.synthW_out1[j]*h[1] +
                    w.synthW_out0[j]*h[2] + w.synthW_out1[j]*h[3] +
                    w.synthW_out0[j]*h[4] + w.synthW_out1[j]*h[5] +
                    w.synthBias_out[j];
        out4[j] = 1.0f / (1.0f + std::exp(-val));
    }
}

void initWeights(Weights& w) {
    memset(&w, 0, sizeof(Weights));
    
    // IFNet-Lite hidden weights (8 units)
    float h0[]={1.12f,-0.31f,0.48f,0.86f}, h1[]={-0.49f,0.84f,0.24f,-0.29f};
    float h2[]={0.34f,0.27f,-0.44f,0.75f}, h3[]={0.58f,-0.72f,0.18f,0.31f};
    float h4[]={-0.27f,0.41f,0.95f,-0.33f}, h5[]={0.73f,0.11f,-0.29f,0.63f};
    float h6[]={0.43f,-0.55f,0.71f,-0.21f}, h7[]={-0.62f,0.38f,0.15f,0.47f};
    memcpy(w.mlpW_h0,h0,16); memcpy(w.mlpW_h1,h1,16); memcpy(w.mlpW_h2,h2,16);
    memcpy(w.mlpW_h3,h3,16); memcpy(w.mlpW_h4,h4,16); memcpy(w.mlpW_h5,h5,16);
    memcpy(w.mlpW_h6,h6,16); memcpy(w.mlpW_h7,h7,16);
    
    // Output weights (4 vectors)
    float o0[]={0.10f,-0.05f,0.02f,0.07f}, o1[]={-0.03f,0.01f,0.05f,-0.04f};
    float o2[]={0.0f,0.03f,-0.02f,0.06f}, o3[]={0.05f,-0.02f,0.04f,-0.01f};
    memcpy(w.mlpW_out0,o0,16); memcpy(w.mlpW_out1,o1,16);
    memcpy(w.mlpW_out2,o2,16); memcpy(w.mlpW_out3,o3,16);
    
    // Hidden biases
    float bh0[]={0.03f,-0.01f,0.02f,0.04f}, bh1[]={-0.02f,0.01f,0.02f,-0.01f};
    memcpy(w.mlpBias_h0,bh0,16); memcpy(w.mlpBias_h1,bh1,16);
    
    // Output biases
    float bo0[]={-0.02f,0.01f,0.0f,0.0f}, bo1[]={0.0f,0.0f,0.0f,0.0f};
    float bo2[]={0.0f,0.0f,0.0f,0.0f}, bo3[]={0.0f,0.0f,-2.0f,0.0f};
    memcpy(w.mlpBias_out0,bo0,16); memcpy(w.mlpBias_out1,bo1,16);
    memcpy(w.mlpBias_out2,bo2,16); memcpy(w.mlpBias_out3,bo3,16);
    
    // Base weights
    float bw1[]={0.15f,0.10f,0.10f,0.20f}, bw2[]={0.10f,0.10f,0.15f,0.10f}, bw3[]={0.10f,0.10f,0.10f,0.10f};
    memcpy(w.baseW1,bw1,16); memcpy(w.baseW2,bw2,16); memcpy(w.baseW3,bw3,16);
    
    // FusionNet-Lite synthesis weights: small random init
    float sh0[]={0.15f,-0.12f,0.08f,0.11f}, sh1[]={-0.09f,0.14f,0.06f,-0.07f};
    float sh2[]={0.11f,0.05f,-0.13f,0.10f}, sh3[]={0.07f,-0.10f,0.12f,-0.08f};
    float sh4[]={-0.06f,0.09f,0.14f,-0.05f}, sh5[]={0.10f,-0.08f,0.07f,0.12f};
    memcpy(w.synthW_h0,sh0,16); memcpy(w.synthW_h1,sh1,16); memcpy(w.synthW_h2,sh2,16);
    memcpy(w.synthW_h3,sh3,16); memcpy(w.synthW_h4,sh4,16); memcpy(w.synthW_h5,sh5,16);
    float so0[]={0.08f,-0.05f,0.10f,0.03f}, so1[]={-0.06f,0.07f,0.04f,-0.09f};
    memcpy(w.synthW_out0,so0,16); memcpy(w.synthW_out1,so1,16);
    // Synthesis biases default to 0
    
    w.useCustomWeights = 0.0f;
    
    // Small random perturbation to break symmetry
    srand((unsigned int)time(nullptr));
    float* p = w.params();
    // Only perturb trainable weights, not flag/padding (last 4 floats)
    for (int i = 0; i < Weights::NUM_PARAMS - 4; i++)
        p[i] += ((float)(rand() % 200 - 100) / 10000.0f);
}

void cleanStaticImages(std::vector<std::string>& images) {
    if (images.size() < 3) return;
    std::vector<std::string> cleaned;
    for (size_t i = 1; i + 1 < images.size(); i++) cleaned.push_back(images[i]);
    images = cleaned;
}

void trainModel(HWND hwnd) {
    g_training = true;
    g_progress = 0;
    log("Starting training...");
    
    srand((unsigned int)time(nullptr));
    
    createFolder(g_cfg.outputFolder.c_str());
    createFolder((g_cfg.outputFolder + "\\frames").c_str());
    createFolder((g_cfg.outputFolder + "\\ground_truth").c_str());
    
    std::vector<std::string> vidExts = {"mp4", "mkv", "avi", "mov", "webm"};
    std::vector<std::string> videos = getFiles(g_cfg.inputFolder, vidExts);
    
    std::vector<std::string> imgExts = {"png", "jpg", "jpeg", "bmp", "tga"};
    std::vector<std::string> images = getFiles(g_cfg.inputFolder, imgExts);
    
    if (!videos.empty()) {
        log("Found " + std::to_string(videos.size()) + " videos. Extracting frames...");
        std::string framesDir = g_cfg.outputFolder + "\\extracted_frames";
        createFolder(framesDir.c_str());
        
        for (size_t i = 0; i < videos.size() && g_training; i++) {
            log("Extracting frames from video " + std::to_string(i+1) + "/" + std::to_string(videos.size()) + "...");
            std::string cmd = "ffmpeg -i \"" + videos[i] + "\" -vf fps=30 -qscale:v 2 \"" + framesDir + "\\vid" + std::to_string(i) + "_frame_%04d.jpg\" -y -loglevel error";
            system(cmd.c_str());
        }
        
        // Add newly extracted frames to our list
        std::vector<std::string> newImages = getFiles(framesDir, imgExts);
        images.insert(images.end(), newImages.begin(), newImages.end());
    }
    
    if (images.empty()) {
        log("ERROR: No images or videos found!");
        g_training = false;
        return;
    }
    
    log("Found " + std::to_string(images.size()) + " images");
    
    if (g_cfg.cleanStatic && images.size() > 2) {
        cleanStaticImages(images);
        log("Using " + std::to_string(images.size()) + " frames after cleaning");
    }
    
    initWeights(g_weights);
    
    std::vector<std::vector<float>> features;
    std::vector<std::vector<float>> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    log("Extracting features from images...");
    
    int numTriplets = 0;
    for (size_t i = 0; i + 2 * g_cfg.stride < images.size(); i++) {
        numTriplets++;
    }
    
    int maxTriplets = 500; // Cap to keep RIFE generation reasonable
    
    // Compute stride to sample evenly across the dataset
    int sampleStride = 1;
    if (numTriplets > maxTriplets) {
        sampleStride = numTriplets / maxTriplets;
        numTriplets = maxTriplets;
    }

    // Pre-build triplet info (paths for img1, mid, img3, gt)
    struct TripletInfo {
        std::string img1_path;
        std::string img2_path; // original middle frame
        std::string img3_path;
        std::string gt_path;
    };
    std::vector<TripletInfo> triplets;
    {
        int count = 0;
        for (size_t i = 0; i + 2 * g_cfg.stride < images.size() && count < numTriplets; i += sampleStride, count++) {
            TripletInfo t;
            t.img1_path = images[i];
            t.img2_path = images[i + g_cfg.stride];
            t.img3_path = images[i + 2 * g_cfg.stride];
            t.gt_path   = g_cfg.outputFolder + "\\ground_truth\\gt_" + std::to_string(count) + ".jpg";
            triplets.push_back(t);
        }
    }
    numTriplets = (int)triplets.size();

    // Generate RIFE ground truths only for triplets that don't already have a GT file
    if (g_cfg.useRIFE && !triplets.empty()) {
        // Check which GT files already exist
        std::vector<TripletInfo> missingGT;
        int existingCount = 0;
        for (auto& t : triplets) {
            if (fileExists(t.gt_path)) {
                existingCount++;
            } else {
                missingGT.push_back(t);
            }
        }

        if (existingCount > 0) {
            log("Found " + std::to_string(existingCount) + "/" + std::to_string(numTriplets) + " existing Ground Truth frames, reusing them.");
        }

        if (!missingGT.empty()) {
            log("Generating " + std::to_string(missingGT.size()) + " missing Ground Truth frames using RIFE (batch mode)...");

            std::string rifeScript = "tools\\rife\\generate_gt.py";
            if (!fileExists(rifeScript)) {
                rifeScript = "..\\..\\..\\tools\\rife\\generate_gt.py";
            }

            // Write manifest file (only missing GT entries)
            std::string manifestPath = g_cfg.outputFolder + "\\rife_manifest.txt";
            {
                std::ofstream mf(manifestPath);
                for (auto& t : missingGT) {
                    mf << t.img1_path << "|" << t.img3_path << "|" << t.gt_path << "\n";
                }
            }

            std::string rifeCmd = "python \"" + rifeScript + "\" --batch \"" + manifestPath + "\" 2>&1";
            log("Running: " + rifeCmd);
            int ret = system(rifeCmd.c_str());
            if (ret != 0) {
                log("WARNING: RIFE batch command returned non-zero exit code: " + std::to_string(ret));
            }
            log("RIFE batch generation complete.");
        } else {
            log("All " + std::to_string(numTriplets) + " Ground Truth frames already exist. Skipping RIFE generation.");
        }
    }

    // Motion-aware targets for IFNet-Lite training: [residualX, residualY, occlusion, quality]
    std::vector<std::array<float, 4>> motionTargets;
    
    int processed = 0;
    for (int ti = 0; ti < numTriplets && g_training; ti++) {
        auto& t = triplets[ti];

        unsigned char* gt_img = nullptr;
        int w, h, channels;
        
        if (g_cfg.useRIFE) {
            gt_img = stbi_load(t.gt_path.c_str(), &w, &h, &channels, 1); // RIFE generated frame
        } else {
            log("Using original middle frame as Ground Truth " + std::to_string(ti + 1) + "/" + std::to_string(numTriplets) + "...");
            gt_img = stbi_load(t.img2_path.c_str(), &w, &h, &channels, 1); // Original middle frame
        }
        
        unsigned char* img1 = stbi_load(t.img1_path.c_str(), &w, &h, &channels, 1); // Load as grayscale
        unsigned char* img2 = stbi_load(t.img2_path.c_str(), &w, &h, &channels, 1); // Original middle frame
        unsigned char* img3 = stbi_load(t.img3_path.c_str(), &w, &h, &channels, 1);
        
        if (!img1) log("Failed to load img1: " + t.img1_path + " Reason: " + (stbi_failure_reason() ? stbi_failure_reason() : "Unknown"));
        if (!img2) log("Failed to load img2: " + t.img2_path + " Reason: " + (stbi_failure_reason() ? stbi_failure_reason() : "Unknown"));
        if (!img3) log("Failed to load img3: " + t.img3_path + " Reason: " + (stbi_failure_reason() ? stbi_failure_reason() : "Unknown"));
        if (!gt_img) log("Failed to load gt_img: " + t.gt_path + " Reason: " + (stbi_failure_reason() ? stbi_failure_reason() : "Unknown"));

        if (img1 && img2 && img3 && gt_img) {
            // === Feature & Target Extraction with Motion Awareness ===
            std::vector<float> feat(12, 0.0f);
            std::vector<float> tgt(12, 0.0f);
            
            int cellW = w / 4;
            int cellH = h / 3;
            
            // Per-cell: block matching + warp error + motion targets
            float globalWarpErr = 0, globalBlendErr = 0;
            int globalCount = 0;
            float globalResX = 0, globalResY = 0;
            int occludedCells = 0;
            
            for (int cy = 0; cy < 3; cy++) {
                for (int cx = 0; cx < 4; cx++) {
                    int cellIdx = cy * 4 + cx;
                    int y0 = cy * cellH, y1 = std::min((cy + 1) * cellH, h);
                    int x0 = cx * cellW, x1 = std::min((cx + 1) * cellW, w);
                    
                    // --- Gradient energy feature (matches shader CNN features) ---
                    float diffSum = 0;
                    int diffCount = 0;
                    for (int y = y0; y < y1; y += 4) {
                        for (int x = x0; x < x1; x += 4) {
                            int idx = y * w + x;
                            float d1 = std::abs((float)img2[idx] - (float)img1[idx]);
                            float d2 = std::abs((float)img3[idx] - (float)img2[idx]);
                            diffSum += (d1 + d2) / 2.0f;
                            diffCount++;
                        }
                    }
                    feat[cellIdx] = (diffCount > 0) ? (diffSum / diffCount) / 255.0f : 0.0f;
                    
                    // --- Block matching: find best integer motion from img1->img3 in this cell ---
                    // Search ±8 pixels, measure SAD in cell center region
                    const int searchR = 8;
                    int bestDx = 0, bestDy = 0;
                    float bestSAD = 1e30f;
                    int patchY0 = y0 + cellH / 4, patchY1 = y1 - cellH / 4;
                    int patchX0 = x0 + cellW / 4, patchX1 = x1 - cellW / 4;
                    if (patchY0 >= patchY1) { patchY0 = y0; patchY1 = y1; }
                    if (patchX0 >= patchX1) { patchX0 = x0; patchX1 = x1; }
                    
                    for (int dy = -searchR; dy <= searchR; dy += 2) {
                        for (int dx = -searchR; dx <= searchR; dx += 2) {
                            float sad = 0;
                            int cnt = 0;
                            for (int y = patchY0; y < patchY1; y += 4) {
                                for (int x = patchX0; x < patchX1; x += 4) {
                                    int sx = std::clamp(x + dx, 0, w - 1);
                                    int sy = std::clamp(y + dy, 0, h - 1);
                                    sad += std::abs((float)img1[sy * w + sx] - (float)img3[y * w + x]);
                                    cnt++;
                                }
                            }
                            if (cnt > 0) sad /= cnt;
                            if (sad < bestSAD) { bestSAD = sad; bestDx = dx; bestDy = dy; }
                        }
                    }
                    // Refine to ±1 around best coarse
                    int coarseDx = bestDx, coarseDy = bestDy;
                    for (int dy = coarseDy - 1; dy <= coarseDy + 1; dy++) {
                        for (int dx = coarseDx - 1; dx <= coarseDx + 1; dx++) {
                            float sad = 0;
                            int cnt = 0;
                            for (int y = patchY0; y < patchY1; y += 4) {
                                for (int x = patchX0; x < patchX1; x += 4) {
                                    int sx = std::clamp(x + dx, 0, w - 1);
                                    int sy = std::clamp(y + dy, 0, h - 1);
                                    sad += std::abs((float)img1[sy * w + sx] - (float)img3[y * w + x]);
                                    cnt++;
                                }
                            }
                            if (cnt > 0) sad /= cnt;
                            if (sad < bestSAD) { bestSAD = sad; bestDx = dx; bestDy = dy; }
                        }
                    }
                    
                    // --- Warp img1 by half-motion toward img2 time, compute warp error vs GT ---
                    float halfDx = bestDx * 0.5f, halfDy = bestDy * 0.5f;
                    float warpErr = 0, blendErr = 0;
                    float residualAccX = 0, residualAccY = 0;
                    int warpCount = 0;
                    
                    for (int y = y0; y < y1; y += 4) {
                        for (int x = x0; x < x1; x += 4) {
                            // Warped pixel from img1
                            int wx = std::clamp((int)(x + halfDx + 0.5f), 0, w - 1);
                            int wy = std::clamp((int)(y + halfDy + 0.5f), 0, h - 1);
                            float warped = (float)img1[wy * w + wx];
                            float gt = (float)gt_img[y * w + x];
                            float blended = ((float)img1[y * w + x] + (float)img3[y * w + x]) * 0.5f;
                            
                            warpErr += std::abs(warped - gt);
                            blendErr += std::abs(blended - gt);
                            
                            // Compute gradient of warp error w.r.t. motion (finite diff)
                            // Which direction should we shift to reduce |warped - gt|?
                            int wxP = std::clamp(wx + 1, 0, w - 1);
                            int wxM = std::clamp(wx - 1, 0, w - 1);
                            int wyP = std::clamp(wy + 1, 0, h - 1);
                            int wyM = std::clamp(wy - 1, 0, h - 1);
                            float errXp = std::abs((float)img1[wy * w + wxP] - gt);
                            float errXm = std::abs((float)img1[wy * w + wxM] - gt);
                            float errYp = std::abs((float)img1[wyP * w + wx] - gt);
                            float errYm = std::abs((float)img1[wyM * w + wx] - gt);
                            // Negative gradient direction = correction that reduces error
                            residualAccX += (errXm - errXp) * 0.5f;
                            residualAccY += (errYm - errYp) * 0.5f;
                            warpCount++;
                        }
                    }
                    
                    if (warpCount > 0) {
                        warpErr /= warpCount;
                        blendErr /= warpCount;
                        residualAccX /= warpCount;
                        residualAccY /= warpCount;
                    }
                    
                    // Target = how much this cell needs attention (warp error as importance)
                    tgt[cellIdx] = warpErr / 255.0f;
                    
                    // Accumulate global motion stats
                    globalWarpErr += warpErr;
                    globalBlendErr += blendErr;
                    globalCount++;
                    // Accumulate residual direction (normalize to ±0.5 sub-pixel range)
                    float resScale = 0.5f / (255.0f + 1e-6f);
                    globalResX += residualAccX * resScale;
                    globalResY += residualAccY * resScale;
                    // Cell is "occluded" if warp error is much worse than blend error (MV is wrong)
                    if (warpErr > blendErr * 1.5f && warpErr > 10.0f) occludedCells++;
                }
            }
            
            features.push_back(feat);
            targets.push_back(tgt);
            
            // === Motion-aware targets for IFNet-Lite (residual, occlusion, quality) ===
            std::array<float, 4> mt = {0, 0, 0, 0.5f};
            if (globalCount > 0) {
                // Average residual correction (clamped to \u00b10.5 sub-pixel range)
                mt[0] = std::clamp(globalResX / globalCount, -0.5f, 0.5f);
                mt[1] = std::clamp(globalResY / globalCount, -0.5f, 0.5f);
                mt[2] = (float)occludedCells / (float)globalCount;
                // Quality = how much warping improves over simple blending (0=bad, 1=perfect)
                float avgWarpErr = globalWarpErr / globalCount;
                float avgBlendErr = globalBlendErr / globalCount;
                mt[3] = std::clamp(1.0f - avgWarpErr / (avgBlendErr + 1.0f), 0.0f, 1.0f);
            }
            motionTargets.push_back(mt);
        }
        
        if (img1) stbi_image_free(img1);
        if (img2) stbi_image_free(img2);
        if (img3) stbi_image_free(img3);
        if (gt_img) stbi_image_free(gt_img);
        
        g_progress = (int)((ti + 1) * 100 / numTriplets);
        processed++;
    }
    
    if (!g_training) {
        log("Training cancelled during feature extraction");
        return;
    }
    
    if (features.empty()) {
        log("ERROR: Failed to extract features from images!");
        g_training = false;
        return;
    }
    
    log("Extracted features from " + std::to_string(features.size()) + " image triplets (with motion targets)");
    
    // Normalize targets to sum-to-1 attention distributions
    for (auto& t : targets) {
        float sum = 1e-6f;
        for (float v : t) sum += std::max(0.0f, v);
        for (float& v : t) v = std::max(0.0f, v) / sum;
    }
    
    log("Training " + std::to_string(Weights::NUM_PARAMS) + " parameters for " + std::to_string(g_cfg.epochs) + " epochs...");
    
    // Loss function: MSE over attention weights + extra outputs regularization + synthesis regularization
    auto computeLoss = [&](const Weights& weights) -> float {
        float totalLoss = 0;
        for (size_t i = 0; i < features.size(); i++) {
            // Attention weight loss (main objective)
            float out[12];
            forward(weights, features[i].data(), out);
            for (int j = 0; j < 12; j++) {
                float d = out[j] - targets[i][j];
                totalLoss += d * d;
            }
            
            // === Motion-aware loss: train IFNet outputs against real warp targets ===
            float motRes[2], occ, qual;
            forwardExtra(weights, features[i].data(), motRes, &occ, &qual);
            
            if (i < motionTargets.size()) {
                const auto& mt = motionTargets[i];
                // Motion residual: train toward computed ideal sub-pixel correction
                float dRx = motRes[0] - mt[0];
                float dRy = motRes[1] - mt[1];
                totalLoss += 0.05f * (dRx * dRx + dRy * dRy);
                
                // Occlusion: train to predict actual occlusion fraction
                float dOcc = occ - mt[2];
                totalLoss += 0.03f * (dOcc * dOcc);
                
                // Quality: train to predict warp reliability
                float dQual = qual - mt[3];
                totalLoss += 0.03f * (dQual * dQual);
            } else {
                // Fallback: light regularization if motion targets missing
                totalLoss += 0.005f * (motRes[0]*motRes[0] + motRes[1]*motRes[1]);
                totalLoss += 0.002f * (occ - 0.3f) * (occ - 0.3f);
                totalLoss += 0.002f * (qual - 0.5f) * (qual - 0.5f);
            }
            
            // Synthesis MLP: train occlusion selection toward warp occlusion signal
            float synth[4];
            synthForward(weights, features[i].data(), synth);
            if (i < motionTargets.size()) {
                // Occlusion select: when occluded, prefer curr (1.0); when visible, neutral (0.5)
                float synthTarget = 0.5f + motionTargets[i][2] * 0.4f;
                totalLoss += 0.02f * (synth[0] - synthTarget) * (synth[0] - synthTarget);
            } else {
                totalLoss += 0.01f * (synth[0] - 0.5f) * (synth[0] - 0.5f);
            }
        }
        const float* p = weights.params();
        float reg = 0;
        for (int i = 0; i < Weights::NUM_PARAMS - 4; i++) reg += p[i] * p[i]; // exclude flag+pad
        return totalLoss / features.size() + 0.0001f * reg;
    };
    
    // Reset progress for the training phase
    g_progress = 0;
    auto trainStart = std::chrono::steady_clock::now();
    float bestLoss = computeLoss(g_weights);
    Weights bestWeights = g_weights;
    
    for (int epoch = 0; epoch < g_cfg.epochs && g_training; epoch++) {
        float* p = g_weights.params();
        
        // Numerical gradient descent over trainable parameters (exclude flag+padding)
        constexpr float eps = 1e-4f;
        const int trainableParams = Weights::NUM_PARAMS - 4; // exclude useCustomWeights + pad
        for (int pi = 0; pi < trainableParams && g_training; pi++) {
            float orig = p[pi];
            p[pi] = orig + eps;
            float lossPlus = computeLoss(g_weights);
            p[pi] = orig - eps;
            float lossMinus = computeLoss(g_weights);
            p[pi] = orig; // restore
            
            float grad = (lossPlus - lossMinus) / (2.0f * eps);
            p[pi] -= g_cfg.learningRate * grad;
        }
        
        // Clamp base weights to valid range
        for (int j = 0; j < 4; j++) {
            g_weights.baseW1[j] = std::clamp(g_weights.baseW1[j], 0.01f, 1.0f);
            g_weights.baseW2[j] = std::clamp(g_weights.baseW2[j], 0.01f, 1.0f);
            g_weights.baseW3[j] = std::clamp(g_weights.baseW3[j], 0.01f, 1.0f);
        }
        
        float loss = computeLoss(g_weights);
        if (loss < bestLoss) {
            bestLoss = loss;
            bestWeights = g_weights;
        }
        
        g_progress = (int)((epoch + 1) * 100 / g_cfg.epochs);
        
        if (epoch % 25 == 0 || epoch == g_cfg.epochs - 1) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - trainStart).count();
            char buf[256];
            snprintf(buf, sizeof(buf), "Epoch %d/%d | Loss: %.6f | Best: %.6f | %.1fs",
                     epoch + 1, g_cfg.epochs, loss, bestLoss, elapsed);
            log(buf);
        }
        
        // Small sleep every 10 epochs to keep UI responsive
        if (epoch % 10 == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Use best weights found during training
    g_weights = bestWeights;
    
    bool wasCancelled = !g_training;
    
    auto trainEnd = std::chrono::steady_clock::now();
    float totalTime = std::chrono::duration<float>(trainEnd - trainStart).count();
    
    if (wasCancelled) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Training stopped early (%.1fs, best loss: %.6f). Saving best weights...", totalTime, bestLoss);
        log(buf);
    }
    
    std::string outPath = g_cfg.outputFolder + "\\weights_trained.json";
    std::ofstream file(outPath);
    if (file.is_open()) {
        auto writeArr = [&](const char* name, const float* arr, bool last = false) {
            file << "  \"" << name << "\": [" << arr[0] << ", " << arr[1] << ", " << arr[2] << ", " << arr[3] << "]" << (last ? "\n" : ",\n");
        };
        file << "{\n";
        // IFNet-Lite hidden weights (8)
        writeArr("mlpW_h0", g_weights.mlpW_h0);
        writeArr("mlpW_h1", g_weights.mlpW_h1);
        writeArr("mlpW_h2", g_weights.mlpW_h2);
        writeArr("mlpW_h3", g_weights.mlpW_h3);
        writeArr("mlpW_h4", g_weights.mlpW_h4);
        writeArr("mlpW_h5", g_weights.mlpW_h5);
        writeArr("mlpW_h6", g_weights.mlpW_h6);
        writeArr("mlpW_h7", g_weights.mlpW_h7);
        // IFNet-Lite output weights (4)
        writeArr("mlpW_out0", g_weights.mlpW_out0);
        writeArr("mlpW_out1", g_weights.mlpW_out1);
        writeArr("mlpW_out2", g_weights.mlpW_out2);
        writeArr("mlpW_out3", g_weights.mlpW_out3);
        // Hidden biases
        writeArr("mlpBias_h0", g_weights.mlpBias_h0);
        writeArr("mlpBias_h1", g_weights.mlpBias_h1);
        // Output biases
        writeArr("mlpBias_out0", g_weights.mlpBias_out0);
        writeArr("mlpBias_out1", g_weights.mlpBias_out1);
        writeArr("mlpBias_out2", g_weights.mlpBias_out2);
        writeArr("mlpBias_out3", g_weights.mlpBias_out3);
        // Base weights
        writeArr("baseW1", g_weights.baseW1);
        writeArr("baseW2", g_weights.baseW2);
        writeArr("baseW3", g_weights.baseW3);
        // FusionNet-Lite synthesis weights
        writeArr("synthW_h0", g_weights.synthW_h0);
        writeArr("synthW_h1", g_weights.synthW_h1);
        writeArr("synthW_h2", g_weights.synthW_h2);
        writeArr("synthW_h3", g_weights.synthW_h3);
        writeArr("synthW_h4", g_weights.synthW_h4);
        writeArr("synthW_h5", g_weights.synthW_h5);
        writeArr("synthW_out0", g_weights.synthW_out0);
        writeArr("synthW_out1", g_weights.synthW_out1);
        writeArr("synthBias_h0", g_weights.synthBias_h0);
        writeArr("synthBias_h1", g_weights.synthBias_h1);
        writeArr("synthBias_out", g_weights.synthBias_out, true);
        file << "}\n";
        file.close();
        char buf[256];
        snprintf(buf, sizeof(buf), "SUCCESS! Saved to: %s (%.1fs, final loss: %.6f)", outPath.c_str(), totalTime, bestLoss);
        log(buf);
    } else {
        log("ERROR: Could not save to " + outPath);
    }
    
    g_training = false;
}

void drawRoundRect(HDC hdc, int x, int y, int w, int h, int r, COLORREF fill) {
    HRGN hrgn = CreateRoundRectRgn(x, y, x + w, y + h, r, r);
    HBRUSH hBrush = CreateSolidBrush(fill);
    FillRgn(hdc, hrgn, hBrush);
    DeleteObject(hBrush);
    DeleteObject(hrgn);
}

void drawRoundRectBorder(HDC hdc, int x, int y, int w, int h, int r, COLORREF fill, COLORREF border) {
    HRGN hrgn = CreateRoundRectRgn(x, y, x + w, y + h, r, r);
    HBRUSH hBrush = CreateSolidBrush(fill);
    FillRgn(hdc, hrgn, hBrush);
    DeleteObject(hBrush);
    
    HRGN borderRgn = CreateRoundRectRgn(x, y, x + w, y + h, r, r);
    HBRUSH hBorderBrush = CreateSolidBrush(border);
    FrameRgn(hdc, borderRgn, hBorderBrush, 1, 1);
    DeleteObject(hBorderBrush);
    DeleteObject(borderRgn);
    DeleteObject(hrgn);
}

void drawLabel(HDC hdc, int x, int y, const char* text, COLORREF color = TEXT_COLOR, int size = 14, bool bold = false) {
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, color);
    
    HFONT hFont = CreateFont(size, 0, 0, 0, bold ? FW_SEMIBOLD : FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI Variable");
    HFONT hOld = (HFONT)SelectObject(hdc, hFont);
    
    RECT rc = { x, y, x + 400, y + 20 };
    DrawTextA(hdc, text, -1, &rc, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    
    SelectObject(hdc, hOld);
    DeleteObject(hFont);
}

void drawPrimaryButton(HDC hdc, int x, int y, int w, int h, const char* text, bool hover, bool pressed, bool disabled) {
    COLORREF bg = disabled ? RGB(204, 204, 204) : (pressed ? RGB(0, 95, 170) : (hover ? ACCENT_LIGHT : ACCENT_COLOR));
    drawRoundRect(hdc, x, y, w, h, 6, bg);
    
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, disabled ? RGB(150, 150, 150) : RGB(255, 255, 255));
    
    HFONT hFont = CreateFont(14, 0, 0, 0, FW_MEDIUM, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI Variable");
    HFONT hOld = (HFONT)SelectObject(hdc, hFont);
    
    RECT rc = { x, y, x + w, y + h };
    DrawTextA(hdc, text, -1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    
    SelectObject(hdc, hOld);
    DeleteObject(hFont);
}

void drawSecondaryButton(HDC hdc, int x, int y, int w, int h, const char* text, bool hover, bool pressed, bool disabled) {
    COLORREF bg = pressed ? RGB(230, 230, 230) : (hover ? RGB(246, 246, 246) : CARD_COLOR);
    COLORREF border = disabled ? RGB(204, 204, 204) : (hover ? ACCENT_COLOR : BORDER_COLOR);
    drawRoundRectBorder(hdc, x, y, w, h, 6, bg, border);
    
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, disabled ? RGB(150, 150, 150) : TEXT_COLOR);
    
    HFONT hFont = CreateFont(14, 0, 0, 0, FW_MEDIUM, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI Variable");
    HFONT hOld = (HFONT)SelectObject(hdc, hFont);
    
    RECT rc = { x, y, x + w, y + h };
    DrawTextA(hdc, text, -1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    
    SelectObject(hdc, hOld);
    DeleteObject(hFont);
}

void drawProgressBar(HDC hdc, int x, int y, int w, int h, int percent) {
    drawRoundRect(hdc, x, y, w, h, 4, PROGRESS_BG);
    
    if (percent > 0) {
        int barW = (int)((w - 4) * percent / 100.0f);
        if (barW > 0) {
            drawRoundRect(hdc, x + 2, y + 2, barW, h - 4, 3, ACCENT_COLOR);
        }
    }
}

void drawCheckBox(HDC hdc, int x, int y, const char* text, bool checked, bool hover) {
    COLORREF boxBg = checked ? ACCENT_COLOR : (hover ? RGB(240, 240, 240) : CARD_COLOR);
    drawRoundRectBorder(hdc, x, y + 2, 18, 18, 3, boxBg, checked ? ACCENT_COLOR : BORDER_COLOR);
    
    if (checked) {
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(255, 255, 255));
        HFONT hFont = CreateFont(10, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
        HFONT hOldF = (HFONT)SelectObject(hdc, hFont);
        RECT rc = { x, y + 2, x + 18, y + 20 };
        DrawTextA(hdc, "OK", -1, &rc, DT_CENTER | DT_VCENTER);
        SelectObject(hdc, hOldF);
        DeleteObject(hFont);
    }
    
    drawLabel(hdc, x + 26, y + 1, text);
}

HWND hInputEdit, hOutputEdit, hEpochsEdit, hLREdit, hStrideEdit, hLogEdit;
HFONT hUIFont, hLogFont;
int g_hoverBtn = 0;
bool g_pressedBtn = false;
bool g_cleanStatic = true;

void invalidateAll(HWND hwnd) {
    InvalidateRect(hwnd, NULL, FALSE);
}

void redrawWindow(HWND hwnd, HDC hdc) {
    RECT rc;
    GetClientRect(hwnd, &rc);
    
    HBRUSH hBg = CreateSolidBrush(BG_COLOR);
    FillRect(hdc, &rc, hBg);
    DeleteObject(hBg);
    
    HFONT hTitleFont = CreateFont(28, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI Variable");
    HFONT hOld = (HFONT)SelectObject(hdc, hTitleFont);
    SetTextColor(hdc, TEXT_COLOR);
    SetBkMode(hdc, TRANSPARENT);
    RECT titleRc = { 24, 20, 500, 60 };
    DrawTextA(hdc, "True Motion Fidelity", -1, &titleRc, DT_LEFT | DT_TOP | DT_SINGLELINE);
    SelectObject(hdc, hOld);
    DeleteObject(hTitleFont);
    
    drawLabel(hdc, 24, 55, "Training Suite", TEXT_SECONDARY, 14);
    
    int cardY = 80;
    drawRoundRect(hdc, 16, cardY, 508, 140, 8, CARD_COLOR);
    
    drawLabel(hdc, 32, cardY + 16, "Input Folder", TEXT_SECONDARY, 13);
    
    // Input Folder Border
    drawRoundRectBorder(hdc, 32, cardY + 34, 360, 32, 4, INPUT_BG, g_hoverBtn == 1 ? ACCENT_COLOR : BORDER_COLOR);
    drawSecondaryButton(hdc, 400, cardY + 34, 100, 32, "Browse", g_hoverBtn == 1, g_pressedBtn && g_hoverBtn == 1, FALSE);
    
    drawLabel(hdc, 32, cardY + 76, "Output Folder", TEXT_SECONDARY, 13);
    
    // Output Folder Border
    drawRoundRectBorder(hdc, 32, cardY + 94, 360, 32, 4, INPUT_BG, g_hoverBtn == 2 ? ACCENT_COLOR : BORDER_COLOR);
    drawSecondaryButton(hdc, 400, cardY + 94, 100, 32, "Browse", g_hoverBtn == 2, g_pressedBtn && g_hoverBtn == 2, FALSE);
    
    cardY += 148;
    drawRoundRect(hdc, 16, cardY, 508, 110, 8, CARD_COLOR);
    
    drawLabel(hdc, 32, cardY + 16, "Training Settings", TEXT_SECONDARY, 13);
    
    drawLabel(hdc, 32, cardY + 42, "Epochs", TEXT_COLOR, 13);
    // Epochs Border
    drawRoundRectBorder(hdc, 90, cardY + 40, 60, 28, 4, INPUT_BG, BORDER_COLOR);
    
    drawLabel(hdc, 160, cardY + 42, "Learning Rate", TEXT_COLOR, 13);
    // LR Border
    drawRoundRectBorder(hdc, 250, cardY + 40, 60, 28, 4, INPUT_BG, BORDER_COLOR);
    
    drawLabel(hdc, 320, cardY + 42, "Stride", TEXT_COLOR, 13);
    // Stride Border
    drawRoundRectBorder(hdc, 370, cardY + 40, 60, 28, 4, INPUT_BG, BORDER_COLOR);
    
    drawCheckBox(hdc, 32, cardY + 78, "Clean static images", g_cleanStatic, g_hoverBtn == 5);
    drawCheckBox(hdc, 200, cardY + 78, "Use RIFE Distillation", g_cfg.useRIFE, g_hoverBtn == 6);
    
    if (!g_training) {
        drawPrimaryButton(hdc, 400, cardY + 68, 108, 36, "Start Training", g_hoverBtn == 3, g_pressedBtn && g_hoverBtn == 3, FALSE);
    } else {
        drawSecondaryButton(hdc, 400, cardY + 68, 108, 36, "Stop", g_hoverBtn == 4, g_pressedBtn && g_hoverBtn == 4, FALSE);
    }
    
    int progY = cardY + 125;
    drawLabel(hdc, 32, progY, "Progress", TEXT_COLOR, 14, true);
    drawProgressBar(hdc, 32, progY + 24, 436, 8, g_progress.load());
    char pct[16];
    sprintf(pct, "%d%%", g_progress.load());
    drawLabel(hdc, 476, progY + 20, pct, TEXT_SECONDARY, 13);
    
    int logY = progY + 60;
    drawRoundRect(hdc, 16, logY, 508, 130, 8, CARD_COLOR);
    
    drawLabel(hdc, 32, logY + 12, "Output Log", TEXT_SECONDARY, 13);
    
    // Log Border
    RECT logBox = { 28, logY + 32, 496, logY + 112 };
    drawRoundRectBorder(hdc, logBox.left, logBox.top, logBox.right - logBox.left, logBox.bottom - logBox.top, 4, INPUT_BG, BORDER_COLOR);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            hUIFont = CreateFont(14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI Variable");
            hLogFont = CreateFont(12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Cascadia Mono");
            
            hInputEdit = CreateWindowExA(0, "EDIT", "dataset", WS_VISIBLE | WS_CHILD | ES_AUTOHSCROLL | ES_READONLY, 
                40, 122, 344, 16, hwnd, (HMENU)100, NULL, NULL);
            SendMessage(hInputEdit, WM_SETFONT, (WPARAM)hUIFont, TRUE);
            
            hOutputEdit = CreateWindowExA(0, "EDIT", "output", WS_VISIBLE | WS_CHILD | ES_AUTOHSCROLL | ES_READONLY, 
                40, 182, 344, 16, hwnd, (HMENU)101, NULL, NULL);
            SendMessage(hOutputEdit, WM_SETFONT, (WPARAM)hUIFont, TRUE);
            
            hEpochsEdit = CreateWindowExA(0, "EDIT", "500", WS_VISIBLE | WS_CHILD | ES_AUTOHSCROLL | ES_NUMBER, 
                98, 274, 44, 16, hwnd, (HMENU)102, NULL, NULL);
            SendMessage(hEpochsEdit, WM_SETFONT, (WPARAM)hUIFont, TRUE);
            
            hLREdit = CreateWindowExA(0, "EDIT", "0.005", WS_VISIBLE | WS_CHILD | ES_AUTOHSCROLL, 
                258, 274, 44, 16, hwnd, (HMENU)103, NULL, NULL);
            SendMessage(hLREdit, WM_SETFONT, (WPARAM)hUIFont, TRUE);
            
            hStrideEdit = CreateWindowExA(0, "EDIT", "1", WS_VISIBLE | WS_CHILD | ES_AUTOHSCROLL | ES_NUMBER, 
                378, 274, 44, 16, hwnd, (HMENU)105, NULL, NULL);
            SendMessage(hStrideEdit, WM_SETFONT, (WPARAM)hUIFont, TRUE);
            
            hLogEdit = CreateWindowExA(0, "EDIT", "", WS_VISIBLE | WS_CHILD | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
                30, 447, 464, 76, hwnd, (HMENU)104, NULL, NULL);
            SendMessage(hLogEdit, WM_SETFONT, (WPARAM)hLogFont, TRUE);
            
            SetTimer(hwnd, 1, 100, NULL);
            return 0;
        }
        
        case WM_CTLCOLOREDIT:
        case WM_CTLCOLORSTATIC: {
            HDC hdc = (HDC)wParam;
            HWND hCtl = (HWND)lParam;
            if (hCtl == hInputEdit || hCtl == hOutputEdit || hCtl == hEpochsEdit || hCtl == hLREdit || hCtl == hStrideEdit || hCtl == hLogEdit) {
                SetBkMode(hdc, OPAQUE);
                SetBkColor(hdc, INPUT_BG);
                SetTextColor(hdc, TEXT_COLOR);
                static HBRUSH hBrush = CreateSolidBrush(INPUT_BG);
                return (LRESULT)hBrush;
            }
            return DefWindowProc(hwnd, msg, wParam, lParam);
        }
        
        case WM_ERASEBKGND:
            return TRUE;
            
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            RECT rc;
            GetClientRect(hwnd, &rc);
            int width = rc.right - rc.left;
            int height = rc.bottom - rc.top;
            
            HDC memDC = CreateCompatibleDC(hdc);
            HBITMAP memBitmap = CreateCompatibleBitmap(hdc, width, height);
            HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, memBitmap);
            
            redrawWindow(hwnd, memDC);
            
            BitBlt(hdc, 0, 0, width, height, memDC, 0, 0, SRCCOPY);
            
            SelectObject(memDC, oldBitmap);
            DeleteObject(memBitmap);
            DeleteDC(memDC);
            
            EndPaint(hwnd, &ps);
            return 0;
        }
        
        case WM_MOUSEMOVE: {
            int x = LOWORD(lParam);
            int y = HIWORD(lParam);
            
            int oldHover = g_hoverBtn;
            g_hoverBtn = 0;
            
            if (x >= 400 && x <= 500 && y >= 114 && y <= 146) g_hoverBtn = 1;
            else if (x >= 400 && x <= 500 && y >= 174 && y <= 206) g_hoverBtn = 2;
            else if (!g_training && x >= 400 && x <= 508 && y >= 296 && y <= 332) g_hoverBtn = 3;
            else if (g_training && x >= 400 && x <= 508 && y >= 296 && y <= 332) g_hoverBtn = 4;
            else if (x >= 32 && x <= 180 && y >= 306 && y <= 326) g_hoverBtn = 5;
            else if (x >= 200 && x <= 380 && y >= 306 && y <= 326) g_hoverBtn = 6;
            
            if (oldHover != g_hoverBtn) invalidateAll(hwnd);
            
            if (g_pressedBtn && g_hoverBtn == 0) {
                g_pressedBtn = false;
                invalidateAll(hwnd);
            }
            break;
        }
        
        case WM_LBUTTONDOWN: {
            int x = LOWORD(lParam);
            int y = HIWORD(lParam);
            
            if (x >= 400 && x <= 500 && y >= 114 && y <= 146) {
                g_pressedBtn = true;
                g_hoverBtn = 1;
            } else if (x >= 400 && x <= 500 && y >= 174 && y <= 206) {
                g_pressedBtn = true;
                g_hoverBtn = 2;
            } else if (!g_training && x >= 400 && x <= 508 && y >= 296 && y <= 332) {
                g_pressedBtn = true;
                g_hoverBtn = 3;
            } else if (g_training && x >= 400 && x <= 508 && y >= 296 && y <= 332) {
                g_pressedBtn = true;
                g_hoverBtn = 4;
            } else if (x >= 32 && x <= 200 && y >= 306 && y <= 326) {
                g_cleanStatic = !g_cleanStatic;
            } else if (x >= 220 && x <= 380 && y >= 306 && y <= 326) {
                g_cfg.useRIFE = !g_cfg.useRIFE;
            } else {
                SetFocus(hwnd);
            }
            invalidateAll(hwnd);
            break;
        }
        
        case WM_LBUTTONUP: {
            int x = LOWORD(lParam);
            int y = HIWORD(lParam);
            
            if (g_pressedBtn && g_hoverBtn == 1) {
                char path[MAX_PATH] = { 0 };
                BROWSEINFOA bi = { 0 };
                bi.hwndOwner = hwnd;
                bi.lpszTitle = "Select Input Folder";
                bi.ulFlags = BIF_RETURNONLYFSDIRS;
                LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
                if (pidl && SHGetPathFromIDListA(pidl, path)) {
                    SetWindowText(hInputEdit, path);
                }
            } else if (g_pressedBtn && g_hoverBtn == 2) {
                char path[MAX_PATH] = { 0 };
                BROWSEINFOA bi = { 0 };
                bi.hwndOwner = hwnd;
                bi.lpszTitle = "Select Output Folder";
                bi.ulFlags = BIF_RETURNONLYFSDIRS;
                LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
                if (pidl && SHGetPathFromIDListA(pidl, path)) {
                    SetWindowText(hOutputEdit, path);
                }
            } else if (g_pressedBtn && g_hoverBtn == 3 && !g_training) {
                char buf[256];
                GetWindowText(hInputEdit, buf, 255);
                g_cfg.inputFolder = buf;
                GetWindowText(hOutputEdit, buf, 255);
                g_cfg.outputFolder = buf;
                GetWindowText(hEpochsEdit, buf, 255);
                g_cfg.epochs = atoi(buf);
                GetWindowText(hLREdit, buf, 255);
                g_cfg.learningRate = (float)atof(buf);
                GetWindowText(hStrideEdit, buf, 255);
                g_cfg.stride = atoi(buf);
                if (g_cfg.stride < 1) g_cfg.stride = 1;
                g_cfg.cleanStatic = g_cleanStatic;
                
                {
                    std::lock_guard<std::mutex> lock(g_log_mutex);
                    g_log.clear();
                }
                SetWindowText(hLogEdit, "");
                log("=== True Motion Fidelity Training ===");
                log("Input: " + g_cfg.inputFolder);
                log("Output: " + g_cfg.outputFolder);
                log("Epochs: " + std::to_string(g_cfg.epochs) + " | LR: " + std::to_string(g_cfg.learningRate) + " | Stride: " + std::to_string(g_cfg.stride));
                
                std::thread([hwnd]() {
                    trainModel(hwnd);
                }).detach();
            } else if (g_pressedBtn && g_hoverBtn == 4 && g_training) {
                g_training = false;
                log("Stopping...");
            }
            
            g_pressedBtn = false;
            g_hoverBtn = 0;
            invalidateAll(hwnd);
            break;
        }
            
        case WM_TIMER: {
            static int lastProgress = -1;
            static bool lastTraining = false;
            bool needsRedraw = false;
            
            if (g_log_updated) {
                g_log_updated = false;
                std::string allLog;
                {
                    std::lock_guard<std::mutex> lock(g_log_mutex);
                    for (const auto& s : g_log) {
                        allLog += s + "\r\n";
                    }
                }
                SetWindowText(hLogEdit, allLog.c_str());
                SendMessage(hLogEdit, EM_LINESCROLL, 0, g_log.size());
                needsRedraw = true;
            }
            
            if (g_progress.load() != lastProgress) {
                lastProgress = g_progress.load();
                needsRedraw = true;
            }
            
            if (g_training.load() != lastTraining) {
                lastTraining = g_training.load();
                needsRedraw = true;
            }
            
            if (needsRedraw) {
                invalidateAll(hwnd);
            }
            break;
        }
            
        case WM_DESTROY:
            g_training = false;
            if (hUIFont) DeleteObject(hUIFont);
            if (hLogFont) DeleteObject(hLogFont);
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    WNDCLASSA wc = {0};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInst;
    wc.lpszClassName = "TrainingSuiteClass";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    
    RegisterClassA(&wc);
    
    // Calculate window size to fit client area
    RECT rc = { 0, 0, 540, 580 };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX, FALSE);
    
    HWND hwnd = CreateWindowExA(0, "TrainingSuiteClass", "True Motion Fidelity - Training Suite",
        WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top,
        NULL, NULL, hInst, NULL);
        
    if (!hwnd) return 1;
    
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}
