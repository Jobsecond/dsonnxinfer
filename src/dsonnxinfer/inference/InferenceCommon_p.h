

#ifndef DS_ONNX_INFER_INFERENCECOMMON_P_H
#define DS_ONNX_INFER_INFERENCECOMMON_P_H

#include <filesystem>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <dsonnxinfer/dsonnxinfer_global.h>
#include <dsonnxinfer/Status.h>
#include <flowonnx/tensormap.h>

DSONNXINFER_BEGIN_NAMESPACE

struct Segment;
struct DsConfig;
struct DsVocoderConfig;
struct DsDurConfig;
struct DsPitchConfig;
struct DsVarianceConfig;
struct SpeakerEmbed;
struct SpeakerMixCurve;

using InferMap = flowonnx::TensorMap;

InferMap acousticPreprocess(
        const std::unordered_map<std::string, int64_t> &name2token,
        const std::unordered_map<std::string, int64_t> &languages,
        const Segment &dsSegment,
        const DsConfig &dsConfig,
        double frameLength,
        double transpose,
        Status *status = nullptr);

InferMap linguisticPreprocess(
        const std::unordered_map<std::string, int64_t> &name2token,
        const std::unordered_map<std::string, int64_t> &languages,
        const Segment &dsSegment,
        double frameLength,
        bool predictDur,
        Status *status = nullptr);

InferMap durPreprocess(
        const Segment &dsSegment,
        Status *status = nullptr);

InferMap pitchProcess(
        const Segment &dsSegment,
        const DsPitchConfig &dsPitchConfig,
        double frameLength,
        bool predictDur,
        Status *status = nullptr);

InferMap variancePreprocess(
        const Segment &dsSegment,
        const DsVarianceConfig &dsVarianceConfig,
        double frameLength,
        bool predictDur,
        Status *status = nullptr);

std::vector<float> getSpkMix(const SpeakerEmbed &spkEmb, const std::vector<std::string> &speakers, const SpeakerMixCurve &spkMix, double frameLength, int64_t targetLength);

bool readPhonemesFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out);

bool readMultiLangPhonemesFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out);

bool readLangIdFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out);

inline constexpr int64_t getSpeedupFromSteps(int64_t dsSteps) {
    int64_t dsSpeedup = 10;
    if (dsSteps > 0) {
        dsSpeedup = 1000 / dsSteps;
        if (dsSpeedup < 1) {
            dsSpeedup = 1;
        }
        else if (dsSpeedup > 1000) {
            dsSpeedup = 1000;
        }
        while (((1000 % dsSpeedup) != 0) && (dsSpeedup > 1)) {
            --dsSpeedup;
        }
    }
    return dsSpeedup;
}

DSONNXINFER_END_NAMESPACE
#endif //DS_ONNX_INFER_INFERENCECOMMON_P_H
