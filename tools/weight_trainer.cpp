// ============================================================================
// Offline Weight Trainer for True Motion Fidelity Engine
// CPU-based trainer - no GPU needed
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>

struct AttentionWeights {
    float mlpW_h0[4], mlpW_h1[4], mlpW_h2[4], mlpW_h3[4], mlpW_h4[4], mlpW_h5[4];
    float mlpW_out0[4], mlpW_out1[4], mlpW_out2[4];
    float mlpBias_hidden[4], mlpBias_out0[4], mlpBias_out1[4];
    float baseW1[4], baseW2[4], baseW3[4];
};

struct TrainingConfig {
    int numEpochs = 100;
    float learningRate = 0.01f;
    int batchSize = 32;
    std::string outputFile = "weights_trained.json";
};

class OfflineTrainer {
public:
    bool Train(const TrainingConfig& config);
    bool SaveWeights(const std::string& path);
    
private:
    AttentionWeights m_weights = {};
    
    void InitializeWeights();
    void CnnForward(const float energy[12], float gate[12]);
    float ComputeLoss(const float predicted[12], const float target[12]);
    void BackwardPass(const float* features, const float* output, const float* target);
    void UpdateWeights(float lr);
    
    std::vector<std::vector<float>> m_trainingData;
    std::vector<float> m_groundTruth;
    
    float m_gradBaseW1[4] = {0}, m_gradBaseW2[4] = {0}, m_gradBaseW3[4] = {0};
    float m_gradW_h0[4] = {0}, m_gradW_h1[4] = {0}, m_gradW_h2[4] = {0};
    float m_gradW_h3[4] = {0}, m_gradW_h4[4] = {0}, m_gradW_h5[4] = {0};
};

void OfflineTrainer::InitializeWeights() {
    float mlpH0[] = { 1.12f, -0.31f, 0.48f, 0.86f };
    float mlpH1[] = { -0.49f, 0.84f, 0.24f, -0.29f };
    float mlpH2[] = { 0.34f, 0.27f, -0.44f, 0.75f };
    float mlpH3[] = { 0.58f, -0.72f, 0.18f, 0.31f };
    float mlpH4[] = { -0.27f, 0.41f, 0.95f, -0.33f };
    float mlpH5[] = { 0.73f, 0.11f, -0.29f, 0.63f };
    
    memcpy(m_weights.mlpW_h0, mlpH0, 4*sizeof(float));
    memcpy(m_weights.mlpW_h1, mlpH1, 4*sizeof(float));
    memcpy(m_weights.mlpW_h2, mlpH2, 4*sizeof(float));
    memcpy(m_weights.mlpW_h3, mlpH3, 4*sizeof(float));
    memcpy(m_weights.mlpW_h4, mlpH4, 4*sizeof(float));
    memcpy(m_weights.mlpW_h5, mlpH5, 4*sizeof(float));
    
    float mlpO0[] = { 0.10f, -0.05f, 0.02f, 0.07f };
    float mlpO1[] = { -0.03f, 0.01f, 0.05f, -0.04f };
    float mlpO2[] = { 0.00f, 0.03f, -0.02f, 0.06f };
    memcpy(m_weights.mlpW_out0, mlpO0, 4*sizeof(float));
    memcpy(m_weights.mlpW_out1, mlpO1, 4*sizeof(float));
    memcpy(m_weights.mlpW_out2, mlpO2, 4*sizeof(float));
    
    float biasH[] = { 0.03f, -0.01f, 0.02f, 0.04f };
    memcpy(m_weights.mlpBias_hidden, biasH, 4*sizeof(float));
    
    float bW1[] = { 0.15f, 0.1f, 0.1f, 0.2f };
    float bW2[] = { 0.1f, 0.1f, 0.15f, 0.1f };
    float bW3[] = { 0.1f, 0.1f, 0.1f, 0.1f };
    memcpy(m_weights.baseW1, bW1, 4*sizeof(float));
    memcpy(m_weights.baseW2, bW2, 4*sizeof(float));
    memcpy(m_weights.baseW3, bW3, 4*sizeof(float));
}

void OfflineTrainer::CnnForward(const float energy[12], float gate[12]) {
    float total = 0;
    for(int i = 0; i < 12; i++) total += energy[i];
    if(total < 0.0001f) total = 0.0001f;
    
    float x[12];
    for(int i = 0; i < 12; i++) x[i] = std::max(energy[i] / total, 0.0f);
    
    float h[6];
    float* W_h[6] = { m_weights.mlpW_h0, m_weights.mlpW_h1, m_weights.mlpW_h2, 
                      m_weights.mlpW_h3, m_weights.mlpW_h4, m_weights.mlpW_h5 };
    
    for(int i = 0; i < 6; i++) {
        h[i] = 0;
        for(int j = 0; j < 12; j++) {
            h[i] += x[j] * W_h[i][j % 4] * ((j < 4) ? 1.0f : (j < 8) ? -1.0f : 1.0f);
        }
        h[i] += m_weights.mlpBias_hidden[i];
        h[i] = std::max(h[i], 0.0f);
    }
    
    float* W_o[3] = { m_weights.mlpW_out0, m_weights.mlpW_out1, m_weights.mlpW_out2 };
    
    for(int i = 0; i < 4; i++) gate[i] = W_o[0][i] * h[0] + W_o[1][i] * h[2] + W_o[2][i] * h[4];
    for(int i = 0; i < 4; i++) gate[4+i] = W_o[0][i] * h[1] + W_o[1][i] * h[3] + W_o[2][i] * h[5];
    for(int i = 0; i < 4; i++) gate[8+i] = W_o[0][i] * (h[0]+h[1]) + W_o[1][i] * (h[2]+h[3]) + W_o[2][i] * (h[4]+h[5]);
    
    for(int i = 0; i < 12; i++) {
        gate[i] = 1.0f / (1.0f + std::exp(-gate[i] * 1.35f));
        gate[i] = std::max(0.0f, std::min(1.0f, gate[i]));
    }
}

float OfflineTrainer::ComputeLoss(const float predicted[12], const float target[12]) {
    float loss = 0;
    for(int i = 0; i < 12; i++) {
        float diff = predicted[i] - target[i];
        loss += diff * diff;
    }
    return loss / 12.0f;
}

void OfflineTrainer::BackwardPass(const float* features, const float* output, const float* target) {
    float epsilon = 0.001f;
    float originalLoss = ComputeLoss(output, target);
    
    for(int w = 0; w < 4; w++) {
        float orig = m_weights.baseW1[w];
        m_weights.baseW1[w] += epsilon;
        float gate[12];
        CnnForward(features, gate);
        float newLoss = ComputeLoss(gate, target);
        m_gradBaseW1[w] = (newLoss - originalLoss) / epsilon;
        m_weights.baseW1[w] = orig;
        
        orig = m_weights.baseW2[w];
        m_weights.baseW2[w] += epsilon;
        CnnForward(features, gate);
        newLoss = ComputeLoss(gate, target);
        m_gradBaseW2[w] = (newLoss - originalLoss) / epsilon;
        m_weights.baseW2[w] = orig;
        
        orig = m_weights.baseW3[w];
        m_weights.baseW3[w] += epsilon;
        CnnForward(features, gate);
        newLoss = ComputeLoss(gate, target);
        m_gradBaseW3[w] = (newLoss - originalLoss) / epsilon;
        m_weights.baseW3[w] = orig;
    }
    
    float* W_h[6] = { m_weights.mlpW_h0, m_weights.mlpW_h1, m_weights.mlpW_h2, 
                      m_weights.mlpW_h3, m_weights.mlpW_h4, m_weights.mlpW_h5 };
    float* gradW_h[6] = { m_gradW_h0, m_gradW_h1, m_gradW_h2, m_gradW_h3, m_gradW_h4, m_gradW_h5 };
    
    for(int w = 0; w < 6; w++) {
        for(int k = 0; k < 4; k++) {
            float orig = W_h[w][k];
            W_h[w][k] += epsilon;
            float gate[12];
            CnnForward(features, gate);
            float newLoss = ComputeLoss(gate, target);
            gradW_h[w][k] = (newLoss - originalLoss) / epsilon;
            W_h[w][k] = orig;
        }
    }
}

void OfflineTrainer::UpdateWeights(float lr) {
    float* W_h[6] = { m_weights.mlpW_h0, m_weights.mlpW_h1, m_weights.mlpW_h2, 
                      m_weights.mlpW_h3, m_weights.mlpW_h4, m_weights.mlpW_h5 };
    float* gradW_h[6] = { m_gradW_h0, m_gradW_h1, m_gradW_h2, m_gradW_h3, m_gradW_h4, m_gradW_h5 };
    
    for(int i = 0; i < 4; i++) {
        m_weights.baseW1[i] -= lr * m_gradBaseW1[i];
        m_weights.baseW2[i] -= lr * m_gradBaseW2[i];
        m_weights.baseW3[i] -= lr * m_gradBaseW3[i];
        
        m_weights.baseW1[i] = std::max(0.01f, std::min(1.0f, m_weights.baseW1[i]));
        m_weights.baseW2[i] = std::max(0.01f, std::min(1.0f, m_weights.baseW2[i]));
        m_weights.baseW3[i] = std::max(0.01f, std::min(1.0f, m_weights.baseW3[i]));
    }
    
    for(int w = 0; w < 6; w++) {
        for(int k = 0; k < 4; k++) {
            W_h[w][k] -= lr * gradW_h[w][k] * 0.1f;
        }
    }
}

bool OfflineTrainer::Train(const TrainingConfig& config) {
    std::cout << "Generating synthetic training data...\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for(int i = 0; i < 1000; i++) {
        std::vector<float> features(12);
        int sceneType = i % 5;
        
        if(sceneType == 0) {
            for(int j = 0; j < 12; j++) features[j] = 0.1f + dis(gen) * 0.1f;
        } else if(sceneType == 1) {
            for(int j = 0; j < 4; j++) features[j] = 0.8f + dis(gen) * 0.2f;
            for(int j = 4; j < 12; j++) features[j] = 0.1f + dis(gen) * 0.1f;
        } else if(sceneType == 2) {
            for(int j = 0; j < 12; j++) features[j] = 0.4f + dis(gen) * 0.3f;
        } else if(sceneType == 3) {
            for(int j = 0; j < 12; j++) features[j] = 0.5f + sin(j * 0.5f) * 0.3f;
        } else {
            for(int j = 0; j < 8; j++) features[j] = 0.2f + dis(gen) * 0.2f;
            for(int j = 8; j < 12; j++) features[j] = 0.7f + dis(gen) * 0.2f;
        }
        
        m_trainingData.push_back(features);
        
        float gt[12] = {0};
        if(sceneType == 0) {
            for(int j = 0; j < 4; j++) gt[j] = 0.25f;
        } else if(sceneType == 1) {
            gt[0] = 0.5f; gt[1] = 0.3f; gt[2] = 0.1f; gt[3] = 0.1f;
        } else if(sceneType == 2) {
            for(int j = 0; j < 4; j++) gt[j] = 0.25f;
        } else if(sceneType == 3) {
            for(int j = 0; j < 4; j++) gt[j] = 0.25f;
        } else {
            gt[0] = 0.1f; gt[1] = 0.2f; gt[2] = 0.3f; gt[3] = 0.4f;
        }
        
        for(int j = 0; j < 12; j++) m_groundTruth.push_back(gt[j]);
    }
    
    std::cout << "Generated " << m_trainingData.size() << " training samples\n";
    
    InitializeWeights();
    
    std::cout << "Starting training...\n";
    std::cout << "Epochs: " << config.numEpochs << ", LR: " << config.learningRate << "\n";
    
    for(int epoch = 0; epoch < config.numEpochs; epoch++) {
        float totalLoss = 0;
        int numSamples = 0;
        
        std::vector<int> indices(m_trainingData.size());
        for(size_t i = 0; i < indices.size(); i++) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));
        
        for(size_t idx = 0; idx < indices.size(); idx++) {
            int i = indices[idx];
            const float* features = m_trainingData[i].data();
            const float* target = &m_groundTruth[i * 12];
            
            float output[12];
            CnnForward(features, output);
            
            float loss = ComputeLoss(output, target);
            totalLoss += loss;
            numSamples++;
            
            if(numSamples % config.batchSize == 0) {
                BackwardPass(features, output, target);
                UpdateWeights(config.learningRate);
            }
        }
        
        if(epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << "/" << config.numEpochs 
                      << ", Avg Loss: " << (totalLoss / numSamples) << "\n";
        }
    }
    
    std::cout << "Training complete!\n";
    return true;
}

bool OfflineTrainer::SaveWeights(const std::string& path) {
    std::ofstream file(path);
    if(!file.is_open()) {
        std::cerr << "Failed to save weights to " << path << "\n";
        return false;
    }
    
    file << "{\n";
    
    float* W_h[6] = { m_weights.mlpW_h0, m_weights.mlpW_h1, m_weights.mlpW_h2, 
                      m_weights.mlpW_h3, m_weights.mlpW_h4, m_weights.mlpW_h5 };
    for(int i = 0; i < 6; i++) {
        file << "  \"mlpW_h" << i << "\": [" << W_h[i][0] << ", " << W_h[i][1] 
             << ", " << W_h[i][2] << ", " << W_h[i][3] << "]";
        if(i < 5) file << ",";
        file << "\n";
    }
    
    float* W_o[3] = { m_weights.mlpW_out0, m_weights.mlpW_out1, m_weights.mlpW_out2 };
    for(int i = 0; i < 3; i++) {
        file << "  \"mlpW_out" << i << "\": [" << W_o[i][0] << ", " << W_o[i][1] 
             << ", " << W_o[i][2] << ", " << W_o[i][3] << "]";
        if(i < 2) file << ",";
        file << "\n";
    }
    
    file << "  \"mlpBias_hidden\": [" << m_weights.mlpBias_hidden[0] << ", " 
         << m_weights.mlpBias_hidden[1] << ", " << m_weights.mlpBias_hidden[2] 
         << ", " << m_weights.mlpBias_hidden[3] << "],\n";
    file << "  \"mlpBias_out0\": [" << m_weights.mlpBias_out0[0] << ", " 
         << m_weights.mlpBias_out0[1] << ", " << m_weights.mlpBias_out0[2] 
         << ", " << m_weights.mlpBias_out0[3] << "],\n";
    file << "  \"mlpBias_out1\": [" << m_weights.mlpBias_out1[0] << ", " 
         << m_weights.mlpBias_out1[1] << ", " << m_weights.mlpBias_out1[2] 
         << ", " << m_weights.mlpBias_out1[3] << "],\n";
    
    file << "  \"baseW1\": [" << m_weights.baseW1[0] << ", " << m_weights.baseW1[1] 
         << ", " << m_weights.baseW1[2] << ", " << m_weights.baseW1[3] << "],\n";
    file << "  \"baseW2\": [" << m_weights.baseW2[0] << ", " << m_weights.baseW2[1] 
         << ", " << m_weights.baseW2[2] << ", " << m_weights.baseW2[3] << "],\n";
    file << "  \"baseW3\": [" << m_weights.baseW3[0] << ", " << m_weights.baseW3[1] 
         << ", " << m_weights.baseW3[2] << ", " << m_weights.baseW3[3] << "]\n";
    
    file << "}\n";
    
    std::cout << "Weights saved to " << path << "\n";
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== True Motion Fidelity Engine - Offline Weight Trainer ===\n\n";
    
    TrainingConfig config;
    config.numEpochs = 100;
    config.learningRate = 0.01f;
    config.outputFile = "weights_trained.json";
    
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "-e" && i+1 < argc) {
            config.numEpochs = std::stoi(argv[++i]);
        } else if(arg == "-l" && i+1 < argc) {
            config.learningRate = std::stof(argv[++i]);
        } else if(arg == "-o" && i+1 < argc) {
            config.outputFile = argv[++i];
        } else if(arg == "-h" || arg == "--help") {
            std::cout << "Usage: weight_trainer [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -e <epochs>   Number of training epochs (default: 100)\n";
            std::cout << "  -l <lr>      Learning rate (default: 0.01)\n";
            std::cout << "  -o <file>    Output file (default: weights_trained.json)\n";
            return 0;
        }
    }
    
    OfflineTrainer trainer;
    
    if(!trainer.Train(config)) {
        std::cerr << "Training failed\n";
        return 1;
    }
    
    if(!trainer.SaveWeights(config.outputFile)) {
        std::cerr << "Failed to save weights\n";
        return 1;
    }
    
    std::cout << "\nDone! Copy weights_trained.json to your exe folder and use Import.\n";
    return 0;
}
