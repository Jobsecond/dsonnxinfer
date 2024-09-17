

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
        double transpose);

InferMap linguisticPreprocess(
        const std::unordered_map<std::string, int64_t> &name2token,
        const std::unordered_map<std::string, int64_t> &languages,
        const Segment &dsSegment,
        double frameLength);

InferMap linguisticPreprocessNoDurPredict(
        const std::unordered_map<std::string, int64_t> &name2token,
        const std::unordered_map<std::string, int64_t> &languages,
        const Segment &dsSegment,
        double frameLength);

InferMap durPreprocess(const Segment &dsSegment);

InferMap pitchProcess(
        const Segment &dsSegment,
        const DsPitchConfig &dsPitchConfig,
        double frameLength);

InferMap variancePreprocess(const Segment &dsSegment, const DsVarianceConfig &dsVarianceConfig, double frameLength);

std::vector<float> getSpkMix(const SpeakerEmbed &spkEmb, const std::vector<std::string> &speakers, const SpeakerMixCurve &spkMix, double frameLength, int64_t targetLength);

bool readPhonemesFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out);

bool readMultiLangPhonemesFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out);

bool readLangIdFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out);

DSONNXINFER_END_NAMESPACE
#endif //DS_ONNX_INFER_INFERENCECOMMON_P_H
