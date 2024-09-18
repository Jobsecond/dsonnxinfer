
#include "DurationInference.h"

#include <fstream>
#include <utility>
#include <nlohmann/json.hpp>

#include <flowonnx/inference.h>
#include "InferenceCommon_p.h"

DSONNXINFER_BEGIN_NAMESPACE

class DurationInference::Impl {
public:
    Status open() {
        if (dsDurConfig.features & kfMultiLanguage) {
            readMultiLangPhonemesFile(dsDurConfig.phonemes, name2token);
            readLangIdFile(dsDurConfig.languages, languages);
        } else {
            readPhonemesFile(dsDurConfig.phonemes, name2token);
        }

        std::string errorMessage;
        if (!inferenceHandle.open({{dsDurConfig.linguistic, false}, {dsDurConfig.dur, false}}, &errorMessage)) {
            return {Status_ModelLoadError, errorMessage};
        }

        return {Status_Ok, ""};
    }

    void close() {
        inferenceHandle.close();
        dsDurConfig = {};
        name2token.clear();
        languages.clear();
    }

    InferMap infer(const Segment &dsSegment, Status *status) {
        int sampleRate = dsDurConfig.sampleRate;
        int hopSize = dsDurConfig.hopSize;
        double frameLength = 1.0 * hopSize / sampleRate;
        bool predictDur = dsDurConfig.features & kfLinguisticPredictDur;

        auto linguisticInputData = linguisticPreprocess(name2token, languages, dsSegment, frameLength, predictDur);
        auto durInputData = durPreprocess(dsSegment);


        flowonnx::InferenceData dataLinguistic, dataDur;

        dataLinguistic.inputData = std::move(linguisticInputData);
        dataLinguistic.bindings.push_back({1, "encoder_out", "encoder_out", false});
        dataLinguistic.bindings.push_back({1, "x_masks", "x_masks", false});
        dataDur.inputData = std::move(durInputData);
        dataDur.outputNames.emplace_back("ph_dur_pred");

        std::vector dataList{dataLinguistic, dataDur};

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

    DsDurConfig dsDurConfig;
    std::unordered_map<std::string, int64_t> name2token;
    std::unordered_map<std::string, int64_t> languages;
    flowonnx::Inference inferenceHandle;
};

DurationInference::DurationInference(DsDurConfig &&dsDurConfig)
        : IInference(), _impl(std::make_unique<Impl>()) {
    m_type = IT_Duration;

    auto &impl = *_impl;
    impl.dsDurConfig = std::move(dsDurConfig);
}

DurationInference::DurationInference(const DsDurConfig &dsDurConfig)
        : IInference(), _impl(std::make_unique<Impl>()) {
    m_type = IT_Duration;

    auto &impl = *_impl;
    impl.dsDurConfig = dsDurConfig;
}

Status DurationInference::open() {
    auto &impl = *_impl;
    return impl.open();
}

void DurationInference::close() {
    auto &impl = *_impl;
    return impl.close();
}

//InferMap DurationInference::infer(const Segment &dsSegment, Status *status) {
//    auto &impl = *_impl;
//    return impl.infer(dsSegment, status);
//}

bool DurationInference::runInPlace(Segment &dsSegment, Status *status) {
    auto &impl = *_impl;
    auto result = impl.infer(dsSegment, status);
    if (result.empty()) {
        return false;
    }
    if (auto it = result.find("ph_dur_pred"); it != result.end()) {
        const auto &tensor = it->second;
        const float *buffer;
        const auto bufferSize = tensor.getDataBuffer<float>(&buffer);
        const double frameLength = 1.0 * impl.dsDurConfig.hopSize / impl.dsDurConfig.sampleRate;

        size_t begin = 0;
        size_t end;
        // align dur
        for (auto &word : dsSegment.words) {
            if (word.phones.empty()) {
                continue;
            }
            auto phNum = word.phones.size();
            auto wordDur = word.duration();
            end = begin + phNum;
            if (begin >= bufferSize || end > bufferSize) {
                break;
            }
            double predWordDur = std::accumulate(buffer + begin, buffer + end, 0.0);
            const double scaleFactor = wordDur / predWordDur;
            word.phones[0].start = 0;
            for (size_t k = 1; k < word.phones.size(); ++k) {
                word.phones[k].start = buffer[begin + k - 1] * scaleFactor;
            }
            begin = end;
        }
        return true;
    }
    return false;
}

DurationInference::~DurationInference() = default;

DSONNXINFER_END_NAMESPACE

