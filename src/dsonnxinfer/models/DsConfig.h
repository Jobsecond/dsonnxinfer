
#ifndef DS_ONNX_INFER_DSCONFIG_H
#define DS_ONNX_INFER_DSCONFIG_H

#include <cstdint>
#include <vector>
#include <filesystem>

#include <dsonnxinfer/dsonnxinfer_global.h>

#include <dsonnxinfer/SpeakerEmbed.h>

DSONNXINFER_BEGIN_NAMESPACE

enum DsFeature: uint32_t {
    kfParamGender               = 1 << 0,
    kfParamVelocity             = 1 << 1,
    kfParamEnergy               = 1 << 2,
    kfParamBreathiness          = 1 << 3,
    kfParamTension              = 1 << 4,
    kfParamVoicing              = 1 << 5,
    kfShallowDiffusion          = 1 << 6,
    kfContinuousAcceleration    = 1 << 7,
    kfVariableDepth             = 1 << 8,
    kfMultiLanguage             = 1 << 9,
    kfParamExpr                 = 1 << 10,
    kfParamNoteRest             = 1 << 11,
    kfLinguisticPredictDur      = 1 << 12,
    kfSpkEmbed                  = 1 << 13,
};

struct DSONNXINFER_EXPORT DsVocoderConfig {
    std::string name;

    std::filesystem::path model;

    int numMelBins = 128;
    int hopSize = 512;
    int sampleRate = 44100;

    static DsVocoderConfig fromYAML(const std::filesystem::path &dsVocoderConfigPath, bool *ok = nullptr);
};

struct DSONNXINFER_EXPORT DsConfig {
    std::filesystem::path phonemes;
    std::filesystem::path languages;
    std::filesystem::path acoustic;
    std::string vocoder;
    std::vector<std::string> speakers;
    SpeakerEmbed spkEmb;

    int hiddenSize = 256;
    int hopSize = 512;
    int sampleRate = 44100;
    float maxDepth = 0.0f;
    uint32_t features = 0;

    static DsConfig fromYAML(const std::filesystem::path &dsConfigPath, bool *ok = nullptr);
};

struct DSONNXINFER_EXPORT DsDurConfig {
    std::filesystem::path phonemes;
    std::filesystem::path languages;
    std::filesystem::path linguistic;
    std::filesystem::path dur;
    std::vector<std::string> speakers;
    SpeakerEmbed spkEmb;

    int hopSize = 512;
    int sampleRate = 44100;

    uint32_t features = 0;

    static DsDurConfig fromYAML(const std::filesystem::path &dsDurConfigPath, bool *ok = nullptr);
};

struct DSONNXINFER_EXPORT DsVarianceConfig {
    std::filesystem::path phonemes;
    std::filesystem::path languages;
    std::filesystem::path linguistic;
    std::filesystem::path variance;
    std::vector<std::string> speakers;
    SpeakerEmbed spkEmb;

    int hopSize = 512;
    int sampleRate = 44100;

    uint32_t features = 0;

    static DsVarianceConfig fromYAML(const std::filesystem::path &dsVarianceConfigPath, bool *ok = nullptr);
};

struct DSONNXINFER_EXPORT DsPitchConfig {
    std::filesystem::path phonemes;
    std::filesystem::path languages;
    std::filesystem::path linguistic;
    std::filesystem::path pitch;
    std::vector<std::string> speakers;
    SpeakerEmbed spkEmb;

    int hopSize = 512;
    int sampleRate = 44100;

    uint32_t features = 0;

    static DsPitchConfig fromYAML(const std::filesystem::path &dsVarianceConfigPath, bool *ok = nullptr);
};
DSONNXINFER_END_NAMESPACE

#endif //DS_ONNX_INFER_DSCONFIG_H
