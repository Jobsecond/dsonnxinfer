
#include "PitchInference.h"

#include <fstream>
#include <utility>
#include <cstring>
#include <nlohmann/json.hpp>

#include <flowonnx/inference.h>
#include "InferenceCommon_p.h"
#include <dsonnxinfer/Environment.h>

DSONNXINFER_BEGIN_NAMESPACE

class PitchInference::Impl {
public:
    Impl() :
            inferenceHandle("ds_pitch"),
            steps(Environment::instance()->defaultSteps()),
            depth(Environment::instance()->defaultDepth()) {}

    Status open() {
        if (dsPitchConfig.features & kfMultiLanguage) {
            readLangIdFile(dsPitchConfig.languages, languages);
        }
        if (isFileExtJson(dsPitchConfig.phonemes)) {
            readMultiLangPhonemesFile(dsPitchConfig.phonemes, name2token);
        } else {
            readPhonemesFile(dsPitchConfig.phonemes, name2token);
        }

        std::string errorMessage;
        if (!inferenceHandle.open({{dsPitchConfig.linguistic, false}, {dsPitchConfig.pitch, false}}, &errorMessage)) {
            return {Status_ModelLoadError, errorMessage};
        }

        return {Status_Ok, ""};
    }

    void close() {
        inferenceHandle.close();
        dsPitchConfig = {};
        name2token.clear();
        languages.clear();
    }

    InferMap infer(const Segment &dsSegment, Status *status) {
        int sampleRate = dsPitchConfig.sampleRate;
        int hopSize = dsPitchConfig.hopSize;
        double frameLength = 1.0 * hopSize / sampleRate;
        bool predictDur = dsPitchConfig.features & kfLinguisticPredictDur;

        auto linguisticInputData = linguisticPreprocess(name2token, languages, dsSegment, frameLength, predictDur);
        auto pitchInputData = pitchProcess(dsSegment, dsPitchConfig, frameLength, predictDur);

        const int64_t shapeArr = 1;

        if (dsPitchConfig.features & kfContinuousAcceleration) {
            pitchInputData["steps"] = flowonnx::Tensor::create(&steps, 1, &shapeArr, 1);
        } else {
            int64_t speedup = getSpeedupFromSteps(steps);
            pitchInputData["speedup"] = flowonnx::Tensor::create(&speedup, 1, &shapeArr, 1);
        }

        flowonnx::InferenceData dataLinguistic, dataPitch;

        dataLinguistic.inputData = std::move(linguisticInputData);
        dataLinguistic.bindings.push_back({1, "encoder_out", "encoder_out", false});
        if (!predictDur) {
            dataLinguistic.bindings.push_back({1, "ph_dur", "ph_dur", true});
        }
        dataLinguistic.outputNames.emplace_back("x_masks");
        dataPitch.inputData = std::move(pitchInputData);
        dataPitch.outputNames.emplace_back("pitch_pred");

        std::vector dataList{dataLinguistic, dataPitch};

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

    DsPitchConfig dsPitchConfig;
    std::unordered_map<std::string, int64_t> name2token;
    std::unordered_map<std::string, int64_t> languages;
    flowonnx::Inference inferenceHandle;
    float depth;
    int64_t steps;
};

PitchInference::PitchInference(DsPitchConfig &&dsPitchConfig)
        : IInference(), _impl(std::make_unique<Impl>()) {
    m_type = IT_Pitch;

    auto &impl = *_impl;
    impl.dsPitchConfig = std::move(dsPitchConfig);
}

PitchInference::PitchInference(const DsPitchConfig &dsPitchConfig)
        : IInference(), _impl(std::make_unique<Impl>()) {
    m_type = IT_Pitch;

    auto &impl = *_impl;
    impl.dsPitchConfig = dsPitchConfig;
}

Status PitchInference::open() {
    auto &impl = *_impl;
    return impl.open();
}

void PitchInference::close() {
    auto &impl = *_impl;
    return impl.close();
}

void PitchInference::setDepth(float depth) {
    auto &impl = *_impl;
    impl.depth = depth;
}

void PitchInference::setSteps(int64_t steps) {
    auto &impl = *_impl;
    impl.steps = steps;
}

float PitchInference::depth() const {
    auto &impl = *_impl;
    return impl.depth;
}

int64_t PitchInference::steps() const {
    auto &impl = *_impl;
    return impl.steps;
}

//InferMap PitchInference::infer(const Segment &dsSegment, Status *status) {
//    auto &impl = *_impl;
//    return impl.infer(dsSegment, status);
//}

bool PitchInference::runInPlace(Segment &dsSegment, Status *status) {
    auto &impl = *_impl;
    auto result = impl.infer(dsSegment, status);
    if (result.empty()) {
        return false;
    }
    double frameLength = 1.0 * impl.dsPitchConfig.hopSize / impl.dsPitchConfig.sampleRate;
    if (auto it = result.find("pitch_pred"); it != result.end()) {
        const auto &tensor = it->second;
        const float *buffer;
        const auto bufferSize = tensor.getDataBuffer<float>(&buffer);

        // Copy predicted pitch data to original segment pitch parameter (overwrite existing)
        auto &pitchParam = dsSegment.parameters["pitch"];
        pitchParam.sample_curve.samples.resize(bufferSize);
        std::copy(buffer, buffer + bufferSize, pitchParam.sample_curve.samples.begin());
        pitchParam.tag = "pitch";
        pitchParam.sample_curve.timestep = frameLength;
        pitchParam.retake_start = 0;
        pitchParam.retake_end = bufferSize;
        return true;
    }
    return false;
}

PitchInference::~PitchInference() = default;

DSONNXINFER_END_NAMESPACE

