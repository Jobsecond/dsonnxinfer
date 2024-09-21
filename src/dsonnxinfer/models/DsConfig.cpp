#include <fstream>
#include <filesystem>

#include <yaml-cpp/yaml.h>
#include <syscmdline/system.h>

#include "DsConfig.h"

#ifdef _WIN32
#define DS_STRING_CONVERT(x) SysCmdLine::utf8ToWide(x)
#else
#define DS_STRING_CONVERT(x) (x)
#endif

DSONNXINFER_BEGIN_NAMESPACE

DsConfig DsConfig::fromYAML(const std::filesystem::path &dsConfigPath, bool *ok) {
    DsConfig dsConfig;
    dsConfig.features = 0;

    std::ifstream fileStream(dsConfigPath);
    if (!fileStream.is_open()) {
        if (ok) {
            *ok = false;
        }
        return dsConfig;
    }

    auto dsConfigDir = dsConfigPath.parent_path();
    YAML::Node config = YAML::Load(fileStream);
    if (const auto node = config["phonemes"]) {
        auto phonemesFilename = DS_STRING_CONVERT(node.as<std::string>());
        dsConfig.phonemes = (dsConfigDir / phonemesFilename).make_preferred();
    }

    if (const auto node = config["languages"]) {
        auto languagesFilename = DS_STRING_CONVERT(node.as<std::string>());
        dsConfig.languages = (dsConfigDir / languagesFilename).make_preferred();
    }

    if (const auto node = config["acoustic"]) {
        auto acousticFilename = DS_STRING_CONVERT(node.as<std::string>());
        dsConfig.acoustic = (dsConfigDir / acousticFilename).make_preferred();
    }

    if (const auto node = config["vocoder"]) {
        dsConfig.vocoder = node.as<std::string>();
    }

    if (const auto node = config["use_key_shift_embed"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfParamGender;
        }
    }

    if (const auto node = config["use_speed_embed"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfParamVelocity;
        }
    }

    if (const auto node = config["use_energy_embed"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfParamEnergy;
        }
    }

    if (const auto node = config["use_breathiness_embed"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfParamBreathiness;
        }
    }

    if (const auto node = config["use_tension_embed"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfParamTension;
        }
    }

    if (const auto node = config["use_voicing_embed"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfParamVoicing;
        }
    }

    if (const auto node = config["use_shallow_diffusion"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfShallowDiffusion;
        }
    }

    if (const auto node = config["max_depth"]) {
        dsConfig.maxDepth = node.as<float>();
    }

    if (const auto node = config["use_continuous_acceleration"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfContinuousAcceleration;
        }
    }

    if (const auto node = config["use_variable_depth"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfVariableDepth;
        }
    }

    if (const auto node = config["use_lang_id"]) {
        if (node.as<bool>()) {
            dsConfig.features |= kfMultiLanguage;
        }
    }

    if (const auto node = config["speakers"]) {
        dsConfig.features |= kfSpkEmbed;
        dsConfig.speakers = node.as<std::vector<std::string>>();
        dsConfig.spkEmb.loadSpeakers(dsConfig.speakers, dsConfigDir);
    }

    if (ok) {
        *ok = true;
    }
    return dsConfig;
}

DsVocoderConfig DsVocoderConfig::fromYAML(const std::filesystem::path &dsVocoderConfigPath, bool *ok) {
    DsVocoderConfig dsVocoderConfig;

    std::ifstream fileStream(dsVocoderConfigPath);
    if (!fileStream.is_open()) {
        if (ok) {
            *ok = false;
        }
        return dsVocoderConfig;
    }

    auto dsVocoderConfigDir = dsVocoderConfigPath.parent_path();
    YAML::Node config = YAML::Load(fileStream);
    if (const auto node = config["name"]) {
        dsVocoderConfig.name = node.as<std::string>();
    }

    if (const auto node = config["model"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsVocoderConfig.model = (dsVocoderConfigDir / model).make_preferred();
    }

    if (const auto node = config["num_mel_bins"]) {
        dsVocoderConfig.numMelBins = node.as<int>();
    }

    if (const auto node = config["hop_size"]) {
        dsVocoderConfig.hopSize = node.as<int>();
    }

    if (const auto node = config["sample_rate"]) {
        dsVocoderConfig.sampleRate = node.as<int>();
    }

    if (ok) {
        *ok = true;
    }
    return dsVocoderConfig;
}

DsDurConfig DsDurConfig::fromYAML(const std::filesystem::path &dsDurConfigPath, bool *ok) {
    DsDurConfig dsDurConfig;
    dsDurConfig.features = 0;

    std::ifstream fileStream(dsDurConfigPath);
    if (!fileStream.is_open()) {
        if (ok) {
            *ok = false;
        }
        return dsDurConfig;
    }

    auto dsDurConfigDir = dsDurConfigPath.parent_path();
    YAML::Node config = YAML::Load(fileStream);
    if (const auto node = config["phonemes"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsDurConfig.phonemes = (dsDurConfigDir / model).make_preferred();
    }

    if (const auto node = config["languages"]) {
        auto languagesFilename = DS_STRING_CONVERT(node.as<std::string>());
        dsDurConfig.languages = (dsDurConfigDir / languagesFilename).make_preferred();
    }

    if (const auto node = config["linguistic"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsDurConfig.linguistic = (dsDurConfigDir / model).make_preferred();
    }

    if (const auto node = config["dur"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsDurConfig.dur = (dsDurConfigDir / model).make_preferred();
    }

    if (const auto node = config["hop_size"]) {
        dsDurConfig.hopSize = node.as<int>();
    }

    if (const auto node = config["sample_rate"]) {
        dsDurConfig.sampleRate = node.as<int>();
    }

    if (const auto node = config["predict_dur"]) {
        if (node.as<bool>()) {
            dsDurConfig.features |= kfLinguisticPredictDur;
        }
    }

    if (const auto node = config["use_lang_id"]) {
        if (node.as<bool>()) {
            dsDurConfig.features |= kfMultiLanguage;
        }
    }

    if (const auto node = config["speakers"]) {
        dsDurConfig.features |= kfSpkEmbed;
        dsDurConfig.speakers = node.as<std::vector<std::string>>();
        dsDurConfig.spkEmb.loadSpeakers(dsDurConfig.speakers, dsDurConfigDir);
    }

    if (ok) {
        *ok = true;
    }
    return dsDurConfig;
}

DsVarianceConfig DsVarianceConfig::fromYAML(const std::filesystem::path &dsVarianceConfigPath, bool *ok) {
    DsVarianceConfig dsVarianceConfig;
    dsVarianceConfig.features = 0;

    std::ifstream fileStream(dsVarianceConfigPath);
    if (!fileStream.is_open()) {
        if (ok) {
            *ok = false;
        }
        return dsVarianceConfig;
    }

    auto dsVarianceConfigDir = dsVarianceConfigPath.parent_path();
    YAML::Node config = YAML::Load(fileStream);
    if (const auto node = config["phonemes"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsVarianceConfig.phonemes = (dsVarianceConfigDir / model).make_preferred();
    }

    if (const auto node = config["languages"]) {
        auto languagesFilename = DS_STRING_CONVERT(node.as<std::string>());
        dsVarianceConfig.languages = (dsVarianceConfigDir / languagesFilename).make_preferred();
    }

    if (const auto node = config["linguistic"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsVarianceConfig.linguistic = (dsVarianceConfigDir / model).make_preferred();
    }

    if (const auto node = config["variance"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsVarianceConfig.variance = (dsVarianceConfigDir / model).make_preferred();
    }

    if (const auto node = config["hop_size"]) {
        dsVarianceConfig.hopSize = node.as<int>();
    }

    if (const auto node = config["sample_rate"]) {
        dsVarianceConfig.sampleRate = node.as<int>();
    }

    if (const auto node = config["predict_dur"]) {
        if (node.as<bool>()) {
            dsVarianceConfig.features |= kfLinguisticPredictDur;
        }
    }

    if (const auto node = config["predict_energy"]) {
        if (node.as<bool>()) {
            dsVarianceConfig.features |= kfParamEnergy;
        }
    }

    if (const auto node = config["predict_breathiness"]) {
        if (node.as<bool>()) {
            dsVarianceConfig.features |= kfParamBreathiness;
        }
    }

    if (const auto node = config["predict_tension"]) {
        if (node.as<bool>()) {
            dsVarianceConfig.features |= kfParamTension;
        }
    }

    if (const auto node = config["predict_voicing"]) {
        if (node.as<bool>()) {
            dsVarianceConfig.features |= kfParamVoicing;
        }
    }

    if (const auto node = config["speakers"]) {
        dsVarianceConfig.features |= kfSpkEmbed;
        dsVarianceConfig.speakers = node.as<std::vector<std::string>>();
        dsVarianceConfig.spkEmb.loadSpeakers(dsVarianceConfig.speakers, dsVarianceConfigDir);
    }

    if (const auto node = config["use_continuous_acceleration"]) {
        if (node.as<bool>()) {
            dsVarianceConfig.features |= kfContinuousAcceleration;
        }
    }

    if (const auto node = config["use_lang_id"]) {
        if (node.as<bool>()) {
            dsVarianceConfig.features |= kfMultiLanguage;
        }
    }

    if (ok) {
        *ok = true;
    }
    return dsVarianceConfig;
}

DsPitchConfig DsPitchConfig::fromYAML(const std::filesystem::path &dsPitchConfigPath, bool *ok) {
    DsPitchConfig dsPitchConfig;
    dsPitchConfig.features = 0;

    std::ifstream fileStream(dsPitchConfigPath);
    if (!fileStream.is_open()) {
        if (ok) {
            *ok = false;
        }
        return dsPitchConfig;
    }

    auto dsPitchConfigDir = dsPitchConfigPath.parent_path();
    YAML::Node config = YAML::Load(fileStream);
    if (const auto node = config["phonemes"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsPitchConfig.phonemes = (dsPitchConfigDir / model).make_preferred();
    }

    if (const auto node = config["languages"]) {
        auto languagesFilename = DS_STRING_CONVERT(node.as<std::string>());
        dsPitchConfig.languages = (dsPitchConfigDir / languagesFilename).make_preferred();
    }

    if (const auto node = config["linguistic"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsPitchConfig.linguistic = (dsPitchConfigDir / model).make_preferred();
    }

    if (const auto node = config["pitch"]) {
        auto model = DS_STRING_CONVERT(node.as<std::string>());
        dsPitchConfig.pitch = (dsPitchConfigDir / model).make_preferred();
    }

    if (const auto node = config["hop_size"]) {
        dsPitchConfig.hopSize = node.as<int>();
    }

    if (const auto node = config["sample_rate"]) {
        dsPitchConfig.sampleRate = node.as<int>();
    }

    if (const auto node = config["predict_dur"]) {
        if (node.as<bool>()) {
            dsPitchConfig.features |= kfLinguisticPredictDur;
        }
    }

    if (const auto node = config["use_expr"]) {
        if (node.as<bool>()) {
            dsPitchConfig.features |= kfParamExpr;
        }
    }

    if (const auto node = config["use_note_rest"]) {
        if (node.as<bool>()) {
            dsPitchConfig.features |= kfParamNoteRest;
        }
    }

    if (const auto node = config["speakers"]) {
        dsPitchConfig.features |= kfSpkEmbed;
        dsPitchConfig.speakers = node.as<std::vector<std::string>>();
        dsPitchConfig.spkEmb.loadSpeakers(dsPitchConfig.speakers, dsPitchConfigDir);
    }

    if (const auto node = config["use_continuous_acceleration"]) {
        if (node.as<bool>()) {
            dsPitchConfig.features |= kfContinuousAcceleration;
        }
    }

    if (const auto node = config["use_lang_id"]) {
        if (node.as<bool>()) {
            dsPitchConfig.features |= kfMultiLanguage;
        }
    }

    if (ok) {
        *ok = true;
    }
    return dsPitchConfig;
}

DSONNXINFER_END_NAMESPACE