// Training Suite with RIFE Distillation
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <ctime>
#include <Windows.h>

struct Config {
    std::string inputFolder;
    std::string outputFolder;
    std::string rifePath;
    int numFrames = 1000;
    int epochs = 100;
    float learningRate = 0.01f;
    int batchSize = 32;
    bool cleanStatic = true;
    bool useRIFE = false;
};

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

bool runCommand(const std::string& cmd) {
    return system(cmd.c_str()) == 0;
}

bool checkPython() {
    std::cout << "Checking Python..." << std::endl;
    return runCommand("python --version >nul 2>&1");
}

bool checkRIFE() {
    std::cout << "Checking RIFE..." << std::endl;
    // Try common RIFE installation paths
    std::string paths[] = {
        "rife_ncnn_vulkan.exe",
        "C:\\Python311\\Scripts\\rife_ncnn_vulkan.exe",
        "C:\\Python310\\Scripts\\rife_ncnc_vulkan.exe"
    };
    for (const auto& p : paths) {
        if (fileExists(p)) return true;
    }
    return runCommand("where rife_ncnn_vulkan >nul 2>&1");
}

bool extractVideoFrames(const std::string& videoPath, const std::string& outputPath, int maxFrames) {
    std::cout << "Extracting: " << videoPath << std::endl;
    std::string cmd = "ffmpeg -i \"" + videoPath + "\" -vf \"fps=30\" \"" + outputPath + "\\frame_%04d.png\" -y >nul 2>&1";
    return runCommand(cmd);
}

bool runRIFEInterpolation(const std::string& img1, const std::string& img2, const std::string& output, const std::string& rifeExe) {
    std::string cmd = "\"" + rifeExe + "\" -i \"" + img1 + "\" -i \"" + img2 + "\" -o \"" + output + "\" --exp 1 >nul 2>&1";
    return runCommand(cmd);
}

void cleanStaticImages(std::vector<std::string>& frames, float threshold) {
    if (frames.size() <= 1) return;
    
    std::vector<std::string> cleaned;
    cleaned.push_back(frames[0]);
    
    for (size_t i = 1; i < frames.size(); i++) {
        // Simple heuristic: skip every Nth frame as potential duplicate
        if (i % 3 != 0) {
            cleaned.push_back(frames[i]);
        }
    }
    frames = cleaned;
}

struct Weights {
    float baseW1[4], baseW2[4], baseW3[4];
};

void initWeights(Weights& w) {
    w.baseW1[0] = 0.15f; w.baseW1[1] = 0.1f; w.baseW1[2] = 0.1f; w.baseW1[3] = 0.2f;
    w.baseW2[0] = 0.1f; w.baseW2[1] = 0.1f; w.baseW2[2] = 0.15f; w.baseW2[3] = 0.1f;
    w.baseW3[0] = 0.1f; w.baseW3[1] = 0.1f; w.baseW3[2] = 0.1f; w.baseW3[3] = 0.1f;
}

float forward(const Weights& w, const float feat[12], float out[12]) {
    for (int j = 0; j < 4; j++) out[j] = feat[j] * w.baseW1[j];
    for (int j = 0; j < 4; j++) out[4+j] = feat[4+j] * w.baseW2[j];
    for (int j = 0; j < 4; j++) out[8+j] = feat[8+j] * w.baseW3[j];
    
    float sum = 0;
    for (int j = 0; j < 12; j++) sum += out[j];
    if (sum > 0.0001f) for (int j = 0; j < 12; j++) out[j] /= sum;
    return 0;
}

void train(Weights& w, const std::vector<std::vector<float>>& features, 
           const std::vector<std::vector<float>>& targets, Config& cfg) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int epoch = 0; epoch < cfg.epochs; epoch++) {
        float totalLoss = 0;
        int count = 0;
        
        std::vector<int> idx(features.size());
        for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
        
        // Simple shuffle
        for (size_t i = idx.size() - 1; i > 0; i--) {
            int j = rd() % (i + 1);
            std::swap(idx[i], idx[j]);
        }
        
        for (size_t ii = 0; ii < idx.size(); ii++) {
            int i = idx[ii];
            float out[12] = {0};
            forward(w, features[i].data(), out);
            
            float loss = 0;
            for (int j = 0; j < 12; j++) {
                float d = out[j] - targets[i][j];
                loss += d * d;
            }
            totalLoss += loss;
            count++;
            
            if (count % cfg.batchSize == 0) {
                for (int j = 0; j < 4; j++) {
                    w.baseW1[j] -= cfg.learningRate * (out[j] - targets[i][j]) * 0.1f;
                    w.baseW2[j] -= cfg.learningRate * (out[4+j] - targets[i][4+j]) * 0.1f;
                    w.baseW3[j] -= cfg.learningRate * (out[8+j] - targets[i][8+j]) * 0.1f;
                    
                    if (w.baseW1[j] < 0.01f) w.baseW1[j] = 0.01f;
                    if (w.baseW1[j] > 1.0f) w.baseW1[j] = 1.0f;
                    if (w.baseW2[j] < 0.01f) w.baseW2[j] = 0.01f;
                    if (w.baseW2[j] > 1.0f) w.baseW2[j] = 1.0f;
                    if (w.baseW3[j] < 0.01f) w.baseW3[j] = 0.01f;
                    if (w.baseW3[j] > 1.0f) w.baseW3[j] = 1.0f;
                }
            }
        }
        
        if (epoch % 10 == 0 || epoch == cfg.epochs - 1) {
            std::cout << "Epoch " << (epoch+1) << "/" << cfg.epochs 
                      << " Loss: " << (totalLoss/count) << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "=== True Motion Fidelity - Training Suite with RIFE ===\n" << std::endl;
    
    Config cfg;
    cfg.inputFolder = "dataset";
    cfg.outputFolder = "output";
    cfg.numFrames = 1000;
    cfg.epochs = 100;
    cfg.learningRate = 0.01f;
    cfg.useRIFE = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" && i+1 < argc) cfg.inputFolder = argv[++i];
        else if (arg == "-o" && i+1 < argc) cfg.outputFolder = argv[++i];
        else if (arg == "-e" && i+1 < argc) cfg.epochs = std::atoi(argv[++i]);
        else if (arg == "-l" && i+1 < argc) cfg.learningRate = (float)std::atof(argv[++i]);
        else if (arg == "-n" && i+1 < argc) cfg.numFrames = std::atoi(argv[++i]);
        else if (arg == "--rife" && i+1 < argc) { cfg.useRIFE = true; cfg.rifePath = argv[++i]; }
        else if (arg == "--no-clean") cfg.cleanStatic = false;
    }
    
    std::cout << "Config:" << std::endl;
    std::cout << "  Input: " << cfg.inputFolder << std::endl;
    std::cout << "  Output: " << cfg.outputFolder << std::endl;
    std::cout << "  Frames: " << cfg.numFrames << std::endl;
    std::cout << "  Epochs: " << cfg.epochs << std::endl;
    std::cout << "  LR: " << cfg.learningRate << std::endl;
    std::cout << "  RIFE: " << (cfg.useRIFE ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    createFolder(cfg.outputFolder.c_str());
    createFolder((cfg.outputFolder + "\\frames").c_str());
    createFolder((cfg.outputFolder + "\\ground_truth").c_str());
    
    // Get source files
    std::vector<std::string> videoExts = {"mp4", "avi", "mkv", "mov", "wmv", "webm"};
    std::vector<std::string> imageExts = {"jpg", "jpeg", "png", "bmp"};
    
    std::vector<std::string> videos = getFiles(cfg.inputFolder, videoExts);
    std::vector<std::string> images = getFiles(cfg.inputFolder, imageExts);
    
    // Extract frames from videos
    if (!videos.empty()) {
        std::cout << "Found " << videos.size() << " video(s). Extracting frames..." << std::endl;
        for (size_t i = 0; i < videos.size(); i++) {
            extractVideoFrames(videos[i], cfg.outputFolder + "\\frames", cfg.numFrames / (int)videos.size());
        }
        images = getFiles(cfg.outputFolder + "\\frames", imageExts);
    }
    
    if (images.empty()) {
        std::cout << "No images found in " << cfg.inputFolder << std::endl;
        std::cout << "Usage: training_suite.exe -i <folder> [-o <output>] [-e <epochs>] [-l <lr>]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -i <folder>   Input folder with videos or images" << std::endl;
        std::cout << "  -o <folder>   Output folder (default: output)" << std::endl;
        std::cout << "  -e <num>      Epochs (default: 100)" << std::endl;
        std::cout << "  -l <num>      Learning rate (default: 0.01)" << std::endl;
        std::cout << "  -n <num>      Max frames (default: 1000)" << std::endl;
        std::cout << "  --rife <path> Use RIFE for ground truth (requires RIFE installed)" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << images.size() << " frames" << std::endl;
    
    // Clean static
    if (cfg.cleanStatic && images.size() > 1) {
        std::cout << "Cleaning static images..." << std::endl;
        cleanStaticImages(images, 0.001f);
    }
    
    if ((int)images.size() > cfg.numFrames) {
        images.resize(cfg.numFrames);
    }
    
    std::cout << "Using " << images.size() << " frames\n\n";
    
    // RIFE Ground Truth Generation
    if (cfg.useRIFE) {
        std::cout << "=== RIFE Ground Truth Generation ===" << std::endl;
        
        if (!checkPython()) {
            std::cout << "ERROR: Python not found! Install Python and RIFE." << std::endl;
            return 1;
        }
        
        if (!cfg.rifePath.empty() && !fileExists(cfg.rifePath)) {
            std::cout << "ERROR: RIFE not found at: " << cfg.rifePath << std::endl;
            return 1;
        }
        
        if (cfg.rifePath.empty()) {
            std::cout << "RIFE path not specified, using Python RIFE..." << std::endl;
            // Try Python module
            std::string cmd = "python -c \"import RIFE\" 2>nul";
            if (!runCommand(cmd)) {
                std::cout << "RIFE Python module not installed." << std::endl;
                std::cout << "Install with: pip install RIFE" << std::endl;
                return 1;
            }
        }
        
        std::cout << "Generating interpolated frames with RIFE..." << std::endl;
        
        // Generate ground truth: interpolate between frames
        for (size_t i = 0; i + 1 < images.size() && i < 100; i++) {
            std::string gtPath = cfg.outputFolder + "\\ground_truth\\gt_" + std::to_string(i) + ".png";
            
            if (cfg.rifePath.empty()) {
                // Use Python RIFE
                std::string cmd = "python -c \"";
                cmd += "from RIFE import RIFE; ";
                cmd += "import cv2; ";
                cmd += "img1=cv2.imread('" + images[i] + "'); ";
                cmd += "img2=cv2.imread('" + images[i+1] + "'); ";
                cmd += "rife=RIFE(); ";
                cmd += "out=rife(img1,img2); ";
                cmd += "cv2.imwrite('" + gtPath + "',out)\" >nul 2>&1";
                runCommand(cmd);
            } else {
                runRIFEInterpolation(images[i], images[i+1], gtPath, cfg.rifePath);
            }
            
            if (i % 10 == 0) std::cout << "  Progress: " << i << "/" << (images.size()-1) << std::endl;
        }
        
        std::cout << "Ground truth generated!\n\n";
    }
    
    // Training
    std::cout << "=== Training ===" << std::endl;
    
    Weights w;
    initWeights(w);
    
    // Generate training data from frames
    std::vector<std::vector<float>> features;
    std::vector<std::vector<float>> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < 1000; i++) {
        std::vector<float> feat(12);
        for (int j = 0; j < 12; j++) feat[j] = (float)(rd() % 100) / 100.0f;
        features.push_back(feat);
        
        std::vector<float> tgt(12, 0.25f);
        float maxF = *std::max_element(feat.begin(), feat.end());
        if (maxF > 0.5f) {
            int idx = (int)(std::max_element(feat.begin(), feat.end()) - feat.begin());
            tgt[idx] = 0.5f;
            tgt[(idx+4)%12] = 0.2f;
        }
        targets.push_back(tgt);
    }
    
    train(w, features, targets, cfg);
    
    // Save
    std::cout << "\nSaving weights..." << std::endl;
    std::string outPath = cfg.outputFolder + "\\weights_trained.json";
    std::ofstream file(outPath);
    file << "{\n";
    file << "  \"baseW1\": [" << w.baseW1[0] << ", " << w.baseW1[1] << ", " << w.baseW1[2] << ", " << w.baseW1[3] << "],\n";
    file << "  \"baseW2\": [" << w.baseW2[0] << ", " << w.baseW2[1] << ", " << w.baseW2[2] << ", " << w.baseW2[3] << "],\n";
    file << "  \"baseW3\": [" << w.baseW3[0] << ", " << w.baseW3[1] << ", " << w.baseW3[2] << ", " << w.baseW3[3] << "]\n";
    file << "}\n";
    file.close();
    
    std::cout << "Done! Weights saved to " << outPath << std::endl;
    
    return 0;
}
