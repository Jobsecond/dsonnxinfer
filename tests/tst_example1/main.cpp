#include <iostream>
#include <sstream>
#include <fstream>

#include <dsonnxinfer/Environment.h>
#include <dsonnxinfer/DsProject.h>
#include <dsonnxinfer/DsConfig.h>
#include <dsonnxinfer/AcousticInference.h>
#include <dsonnxinfer/DurationInference.h>
#include <dsonnxinfer/PitchInference.h>
#include <dsonnxinfer/VarianceInference.h>

using namespace dsonnxinfer;

enum ReturnCode {
    RESULT_OK = 0,
    RESULT_ENV_LOAD_FAILED,
    RESULT_PROJECT_LOAD_FAILED,
    RESULT_MODEL_LOAD_FAILED,
    RESULT_INFERENCE_FAILED,
    RESULT_PROJECT_SAVE_FAILED,
};

int main(int argc, char *argv[]) {
    std::string errorMessage;

    // Load environment (must do this before inference)
    Environment env;
    if (!env.load("onnxruntime", EP_CUDA, &errorMessage)) {
        std::cout << errorMessage << '\n';
        return RESULT_ENV_LOAD_FAILED;
    }

    // Load input data (json format)
    std::ifstream dsFile(R"(D:\test_dsinfer_data\test_dsinfer_data_0.json)");
    if (!dsFile.is_open()) {
        std::cout << "failed to open project file!\n";
        return RESULT_PROJECT_LOAD_FAILED;
    }
    std::stringstream buffer;
    buffer << dsFile.rdbuf();
    std::string jsonData = buffer.str();

    Status s;

    // Initialize inference segment from json data
    Segment segment = Segment::fromJson(jsonData, &s);

    // Load models
    bool loadDsConfigOk;

    std::string dsConfigPath = R"(D:\OpenUtau\Singers\Junninghua_v1.4.0_DiffSinger_OpenUtau\dsconfig.yaml)";
    auto dsConfig = DsConfig::fromYAML(dsConfigPath, &loadDsConfigOk);

    auto dsVocoderConfigPath = std::filesystem::path(dsConfigPath).parent_path() / "dsvocoder" / "vocoder.yaml";
    auto dsVocoderConfig = DsVocoderConfig::fromYAML(dsVocoderConfigPath, &loadDsConfigOk);

    auto durConfigPath = std::filesystem::path(dsConfigPath).parent_path() / "dsdur" / "dsconfig.yaml";
    auto durConfig = DsDurConfig::fromYAML(durConfigPath, &loadDsConfigOk);

    auto pitchConfigPath = std::filesystem::path(dsConfigPath).parent_path() / "dspitch" / "dsconfig.yaml";
    auto pitchConfig = DsPitchConfig::fromYAML(pitchConfigPath, &loadDsConfigOk);

    auto varianceConfigPath = std::filesystem::path(dsConfigPath).parent_path() / "dsvariance" / "dsconfig.yaml";
    auto varianceConfig = DsVarianceConfig::fromYAML(varianceConfigPath, &loadDsConfigOk);

    if (!loadDsConfigOk) {
        s = {Status_ModelLoadError, "Failed to load config!"};
    }
    if (!s.isOk()) {
        return RESULT_MODEL_LOAD_FAILED;
    }

    DurationInference durationInference(durConfig);
    durationInference.open();

    PitchInference pitchInference(pitchConfig);
    pitchInference.open();

    VarianceInference varianceInference(varianceConfig);
    varianceInference.open();

    AcousticInference acousticInference(dsConfig, dsVocoderConfig);
    acousticInference.open();

    auto trySaveSegment = [&](const Segment &currSegment, const std::string &filename) -> bool {
        auto newProject = segment.toJson(&s);
        if (!s.isOk()) {
            std::cout << "Failed to save segment: " << s.msg << '\n';
            return false;
        }
        std::ofstream outFile(filename);
        outFile << newProject;
        outFile.close();
        return true;
    };

    trySaveSegment(segment, "result_original.json");
    // Run duration inference. The segment will be modified in-place.
    if (!durationInference.runInPlace(segment, &s)) {
        std::cout << "Failed to run duration inference: " << s.msg << '\n';
        return RESULT_INFERENCE_FAILED;
    }
    // Save the modified segment to JSON file
    if (!trySaveSegment(segment, "result_dur.json"))
        return RESULT_PROJECT_SAVE_FAILED;

    // Run pitch inference. The segment will be modified in-place.
    if (!pitchInference.runInPlace(segment, &s)) {
        std::cout << "Failed to run pitch inference: " << s.msg << '\n';
        return RESULT_INFERENCE_FAILED;
    }
    if (!trySaveSegment(segment, "result_pitch.json"))
        return RESULT_PROJECT_SAVE_FAILED;

    // Run variance inference. The segment will be modified in-place.
    if (!varianceInference.runInPlace(segment, &s)) {
        std::cout << "Failed to run variance inference: " << s.msg << '\n';
        return RESULT_INFERENCE_FAILED;
    }
    if (!trySaveSegment(segment, "result_variance.json"))
        return RESULT_PROJECT_SAVE_FAILED;

    // Run acoustic inference and export audio.
    if (!acousticInference.runAndSaveAudio(segment, "test.wav", &s)) {
        std::cout << "Failed to run acoustic inference: " << s.msg << '\n';
        return RESULT_INFERENCE_FAILED;
    }

    durationInference.close();
    pitchInference.close();
    varianceInference.close();
    acousticInference.close();

    return RESULT_OK;
}
