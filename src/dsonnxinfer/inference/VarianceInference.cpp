
#include "VarianceInference.h"

#include <fstream>
#include <utility>
#include <cstring>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>

#include <flowonnx/inference.h>
#include "InferenceCommon_p.h"
#include <dsonnxinfer/Environment.h>

DSONNXINFER_BEGIN_NAMESPACE

class VarianceInference::Impl {
public:
    Impl() :
            inferenceHandle("ds_variance"),
            steps(Environment::instance()->defaultSteps()),
            depth(Environment::instance()->defaultDepth()) {}

    Status open() {
        if (dsVarianceConfig.features & kfMultiLanguage) {
            readLangIdFile(dsVarianceConfig.languages, languages);
        }
        if (isFileExtJson(dsVarianceConfig.phonemes)) {
            readMultiLangPhonemesFile(dsVarianceConfig.phonemes, name2token);
        } else {
            readPhonemesFile(dsVarianceConfig.phonemes, name2token);
        }

        if (dsVarianceConfig.features & kfParamEnergy) {
            expectParamNames.emplace_back("energy_pred");
        }
        if (dsVarianceConfig.features & kfParamBreathiness) {
            expectParamNames.emplace_back("breathiness_pred");
        }
        if (dsVarianceConfig.features & kfParamTension) {
            expectParamNames.emplace_back("tension_pred");
        }
        if (dsVarianceConfig.features & kfParamVoicing) {
            expectParamNames.emplace_back("voicing_pred");
        }
        if (dsVarianceConfig.features & kfParamMouthOpening) {
            expectParamNames.emplace_back("mouth_opening_pred");
        }

        std::string errorMessage;
        if (!inferenceHandle.open({{dsVarianceConfig.linguistic, false}, {dsVarianceConfig.variance, false}}, &errorMessage)) {
            return {Status_ModelLoadError, errorMessage};
        }

        return {Status_Ok, ""};
    }

    void close() {
        inferenceHandle.close();
        dsVarianceConfig = {};
        expectParamNames.clear();
        name2token.clear();
        languages.clear();
    }

    InferMap infer(const Segment &dsSegment, Status *status) {
        int sampleRate = dsVarianceConfig.sampleRate;
        int hopSize = dsVarianceConfig.hopSize;
        double frameLength = 1.0 * hopSize / sampleRate;
        bool predictDur = dsVarianceConfig.features & kfLinguisticPredictDur;

        auto linguisticInputData = linguisticPreprocess(name2token, languages, dsSegment, frameLength, predictDur);
        auto varianceInputData = variancePreprocess(dsSegment, dsVarianceConfig, frameLength, predictDur);

        const int64_t shapeArr = 1;

        if (dsVarianceConfig.features & kfContinuousAcceleration) {
            varianceInputData["steps"] = flowonnx::Tensor::create(&steps, 1, &shapeArr, 1);
        } else {
            int64_t speedup = getSpeedupFromSteps(steps);
            varianceInputData["speedup"] = flowonnx::Tensor::create(&speedup, 1, &shapeArr, 1);
        }

        flowonnx::InferenceData dataLinguistic, dataVariance;

        dataLinguistic.inputData = std::move(linguisticInputData);
        dataLinguistic.bindings.push_back({1, "encoder_out", "encoder_out", false});
        if (!predictDur) {
            dataLinguistic.bindings.push_back({1, "ph_dur", "ph_dur", true});
        }
        dataLinguistic.outputNames.emplace_back("x_masks");
        dataVariance.inputData = std::move(varianceInputData);
        dataVariance.outputNames = expectParamNames;

        std::vector dataList{dataLinguistic, dataVariance};

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

    DsVarianceConfig dsVarianceConfig;
    std::unordered_map<std::string, int64_t> name2token;
    std::unordered_map<std::string, int64_t> languages;
    std::vector<std::string> expectParamNames;
    flowonnx::Inference inferenceHandle;
    float depth;
    int64_t steps;
};

VarianceInference::VarianceInference(DsVarianceConfig &&dsVarianceConfig)
        : IInference(), _impl(std::make_unique<Impl>()) {
    m_type = IT_MultiVariance;

    auto &impl = *_impl;
    impl.dsVarianceConfig = std::move(dsVarianceConfig);
}

VarianceInference::VarianceInference(const DsVarianceConfig &dsVarianceConfig)
        : IInference(), _impl(std::make_unique<Impl>()) {
    m_type = IT_MultiVariance;

    auto &impl = *_impl;
    impl.dsVarianceConfig = dsVarianceConfig;
}

Status VarianceInference::open() {
    auto &impl = *_impl;
    return impl.open();
}

void VarianceInference::close() {
    auto &impl = *_impl;
    return impl.close();
}

void VarianceInference::setDepth(float depth) {
    auto &impl = *_impl;
    impl.depth = depth;
}

void VarianceInference::setSteps(int64_t steps) {
    auto &impl = *_impl;
    impl.steps = steps;
}

float VarianceInference::depth() const {
    auto &impl = *_impl;
    return impl.depth;
}

int64_t VarianceInference::steps() const {
    auto &impl = *_impl;
    return impl.steps;
}

//InferMap VarianceInference::infer(const Segment &dsSegment, Status *status) {
//    auto &impl = *_impl;
//    return impl.infer(dsSegment, status);
//}

bool VarianceInference::runInPlace(Segment &dsSegment, Status *status) {
    auto &impl = *_impl;
    auto result = impl.infer(dsSegment, status);
    if (result.empty()) {
        return false;
    }
    double frameLength = 1.0 * impl.dsVarianceConfig.hopSize / impl.dsVarianceConfig.sampleRate;
    for (const auto &paramName : impl.expectParamNames) {
        const std::string inParam(paramName.c_str(), (std::max)(size_t{0}, paramName.size() - 5));
        if (auto it = result.find(paramName); it != result.end()) {
            const auto &tensor = it->second;
            const float *buffer;
            const auto bufferSize = tensor.getDataBuffer<float>(&buffer);

            // Copy predicted pitch data to original segment pitch parameter (overwrite existing)
            auto &currentParam = dsSegment.parameters[inParam];
            currentParam.sample_curve.samples.resize(bufferSize);
            std::copy(buffer, buffer + bufferSize, currentParam.sample_curve.samples.begin());
            currentParam.retake_start = 0;
            currentParam.retake_end = bufferSize;
            currentParam.sample_curve.timestep = frameLength;
            currentParam.tag = inParam;
        }
    }
    return true;
}

bool VarianceInference::terminate() {
    auto &impl = *_impl;
    return impl.terminate();
}

VarianceInference::~VarianceInference() = default;

DSONNXINFER_END_NAMESPACE

