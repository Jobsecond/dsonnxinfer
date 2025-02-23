
#include "AcousticInference.h"

#include <fstream>
#include <utility>
#include <nlohmann/json.hpp>

#include <flowonnx/inference.h>
#include "InferenceCommon_p.h"
#include <dsonnxinfer/Environment.h>

#ifdef DSONNXINFER_ENABLE_AUDIO_EXPORT
#include <sndfile.hh>
#endif

DSONNXINFER_BEGIN_NAMESPACE

class AcousticInference::Impl {
public:
    Impl() :
            inferenceHandle("ds_acoustic"),
            vocoderPreferCpu(true),
            depth(Environment::instance()->defaultDepth()),
            steps(Environment::instance()->defaultSteps()) {}

    Status open() {
        /*bool loadDsConfigOk, loadDsVocoderConfigOk;
        dsConfig = DsConfig::fromYAML(dsConfigPath, &loadDsConfigOk);
        dsVocoderConfig = DsVocoderConfig::fromYAML(dsVocoderConfigPath, &loadDsVocoderConfigOk);

        if (!loadDsConfigOk && loadDsVocoderConfigOk) {
            return {Status_ModelLoadError, "Failed to load voice database config!"};
        } else if (loadDsConfigOk && !loadDsVocoderConfigOk) {
            return {Status_ModelLoadError, "Failed to load vocoder config!"};
        } else if (!loadDsConfigOk && !loadDsVocoderConfigOk) {
            return {Status_ModelLoadError, "Failed to load voice database and vocoder config!"};
        }*/

        if (dsConfig.features & kfMultiLanguage) {
            readLangIdFile(dsConfig.languages, languages);
        }
        if (isFileExtJson(dsConfig.phonemes)) {
            readMultiLangPhonemesFile(dsConfig.phonemes, name2token);
        } else {
            readPhonemesFile(dsConfig.phonemes, name2token);
        }

        std::string errorMessage;
        if (!inferenceHandle.open({{dsConfig.acoustic, false}, {dsVocoderConfig.model, vocoderPreferCpu}}, &errorMessage)) {
            return {Status_ModelLoadError, errorMessage};
        }

        return {Status_Ok, ""};
    }

    void close() {
        inferenceHandle.close();
        dsConfig = {};
        dsVocoderConfig = {};
        name2token.clear();
        languages.clear();
    }

    InferMap infer(const Segment &dsSegment, Status *status) {
        int sampleRate = dsConfig.sampleRate;
        int hopSize = dsConfig.hopSize;
        double frameLength = 1.0 * hopSize / sampleRate;

        auto inputData = acousticPreprocess(
                name2token, languages, dsSegment, dsConfig, frameLength, 0, status);
        if (inputData.empty()) {
            return {};
        }
        const int64_t shapeArr = 1;

        if (dsConfig.features & kfContinuousAcceleration) {
            inputData["steps"] = flowonnx::Tensor::create(&steps, 1, &shapeArr, 1);
        } else {
            int64_t speedup = getSpeedupFromSteps(steps);
            inputData["speedup"] = flowonnx::Tensor::create(&speedup, 1, &shapeArr, 1);
        }

        if (dsConfig.features & kfVariableDepth) {
            inputData["depth"] = flowonnx::Tensor::create(&depth, 1, &shapeArr, 1);
        } else {
            int64_t dsDepthInt64 = std::llround(depth * 1000);
            if ((dsConfig.features & kfShallowDiffusion) && !(dsConfig.features & kfContinuousAcceleration)) {
                if (dsConfig.maxDepth < 0) {
                    putStatus(status, Status_InferError, "!! ERROR: max_depth is unset or negative in acoustic configuration.");
                    return {};
                }
                dsDepthInt64 = (std::min)(dsDepthInt64, static_cast<int64_t>(dsConfig.maxDepth));
                int64_t speedup = getSpeedupFromSteps(steps);
                // make sure depth can be divided by speedup
                dsDepthInt64 = dsDepthInt64 / speedup * speedup;
            }
            inputData["depth"] = flowonnx::Tensor::create(&dsDepthInt64, 1, &shapeArr, 1);
        }

        flowonnx::InferenceData dataAcoustic, dataVocoder;

        dataAcoustic.inputData = std::move(inputData);
        dataAcoustic.bindings.push_back({1, "mel", "mel", false});
        if (dsVocoderConfig.features & kfPitchControllable) {
            if (const auto toneShift = dsSegment.parameters.find("tone_shift"); toneShift != dsSegment.parameters.end()) {
                auto vocoderF0 = dataAcoustic.inputData["f0"];  // copies the f0 tensor
                {
                    float *vocoderF0Data;
                    const auto vocoderF0Size = vocoderF0.getDataBuffer(&vocoderF0Data);
                    auto pitchShiftCents = toneShift->second.sample_curve.resample(
                        frameLength, static_cast<int64_t>(vocoderF0Size), false);
                    for (size_t i = 0; i < vocoderF0Size; ++i) {
                        // assuming `tone_shift` is in cents
                        vocoderF0Data[i] *= static_cast<float>(std::pow(2.0, pitchShiftCents[i] / 1200.0));
                    }
                }
                dataVocoder.inputData["f0"] = std::move(vocoderF0);
            } else {
                dataAcoustic.bindings.push_back({1, "f0", "f0", true});
            }
        } else {
            dataAcoustic.bindings.push_back({1, "f0", "f0", true});
        }

        dataVocoder.outputNames.emplace_back("waveform");

        std::vector dataList{dataAcoustic, dataVocoder};

        std::string errorMessage;
        auto result = inferenceHandle.run(dataList, &errorMessage);

        if (status) {
            if (result.empty()) {
                status->code = Status_InferError;
                status->msg = std::move(errorMessage);
            } else {
                status->code = Status_Ok;
                status->msg = "";
            }
        }
        return result;
    }

    bool terminate() {
        return inferenceHandle.terminate();
    }

    //std::filesystem::path dsConfigPath;
    //std::filesystem::path dsVocoderConfigPath;
    DsConfig dsConfig;
    DsVocoderConfig dsVocoderConfig;
    std::unordered_map<std::string, int64_t> name2token;
    std::unordered_map<std::string, int64_t> languages;
    flowonnx::Inference inferenceHandle;
    bool vocoderPreferCpu;
    float depth;
    int64_t steps;
};

AcousticInference::AcousticInference(DsConfig &&dsConfig,
                                     DsVocoderConfig &&dsVocoderConfig)
        : IInference(), _impl(std::make_unique<Impl>()) {
    m_type = IT_Acoustic;

    auto &impl = *_impl;
    impl.dsConfig = std::move(dsConfig);
    impl.dsVocoderConfig = std::move(dsVocoderConfig);
}

AcousticInference::AcousticInference(const DsConfig &dsConfig,
                                     const DsVocoderConfig &dsVocoderConfig)
        : IInference(), _impl(std::make_unique<Impl>()) {
    m_type = IT_Acoustic;

    auto &impl = *_impl;
    impl.dsConfig = dsConfig;
    impl.dsVocoderConfig = dsVocoderConfig;
}

Status AcousticInference::open() {
    auto &impl = *_impl;
    return impl.open();
}

void AcousticInference::close() {
    auto &impl = *_impl;
    return impl.close();
}

//InferMap AcousticInference::infer(const Segment &dsSegment, Status *status) {
//    auto &impl = *_impl;
//    return impl.infer(dsSegment, status);
//}

void AcousticInference::setDepth(float depth) {
    auto &impl = *_impl;
    impl.depth = depth;
}

void AcousticInference::setSteps(int64_t steps) {
    auto &impl = *_impl;
    impl.steps = steps;
}

float AcousticInference::depth() const {
    auto &impl = *_impl;
    return impl.depth;
}

int64_t AcousticInference::steps() const {
    auto &impl = *_impl;
    return impl.steps;
}

bool AcousticInference::runAndSaveAudio(
        const Segment &dsSegment,
        const std::filesystem::path &path,
        Status *status) {
#ifdef DSONNXINFER_ENABLE_AUDIO_EXPORT
    auto &impl = *_impl;
    const auto result = impl.infer(dsSegment, status);
    if (result.empty()) {
        return false;
    }
    if (auto it = result.find("waveform"); it != result.end()) {
        const auto &tensor = it->second;
        const float *buffer;
        const auto bufferSize = tensor.getDataBuffer<float>(&buffer);
        const auto filePath =
#ifdef _WIN32
            path.wstring();
#else
            path.string();
#endif
        SndfileHandle audioFile(
                filePath.c_str(),
                SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_FLOAT,
                1,
                impl.dsVocoderConfig.sampleRate);
        auto numFrames = static_cast<sf_count_t>(bufferSize);
        auto numWritten = audioFile.write(buffer, numFrames);
        if (numWritten < numFrames) {
            if (status) {
                status->code = Status_GenericError;
                status->msg = "Failed to write audio file";
            }
            return false;
        }
        return true;
    }
    return false;
#else
    if (status) {
        status->code = Status_GenericError;
        status->msg = "DS Onnx Infer is not built with audio export support.";
    }
    return false;
#endif
}

bool AcousticInference::terminate() {
    auto &impl = *_impl;
    return impl.terminate();
}

AcousticInference::~AcousticInference() = default;

DSONNXINFER_END_NAMESPACE

