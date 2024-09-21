#include "InferenceCommon_p.h"

#include <iterator>
#include <numeric>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <fstream>

#include <nlohmann/json.hpp>

#include <dsonnxinfer/DsConfig.h>
#include <dsonnxinfer/DsProject.h>
#include <dsonnxinfer/SampleCurve.h>
#include <dsonnxinfer/SpeakerEmbed.h>


DSONNXINFER_BEGIN_NAMESPACE

using flowonnx::Tensor;

template<typename T>
Tensor toInferDataInPlace(std::vector<T> &&v);

template<typename T>
Tensor toInferDataAndResizeInPlace(std::vector<T> &&v, int64_t targetLength, T val);

template<typename T_Src, typename T_Dst>
Tensor toInferDataAsType(const std::vector<T_Src> &v);

template<typename T_Src, typename T_Dst>
Tensor toInferDataAsTypeAndResize(const std::vector<T_Src> &v, int64_t targetLength, T_Dst val);

template<typename T>
std::vector<T> fillRestMidiWithNearest(const std::vector<T> &src, T restMidi = 0);

template<typename T>
void fillRestMidiWithNearestInPlace(std::vector<T> &src, T restMidi = 0);


template<typename T>
Tensor toInferDataInPlace(std::vector<T> &&v) {
    int64_t size = v.size();
    int64_t shape[2] = {1, size};
    return Tensor::create(v.data(), v.size(), shape, 2);
}

template<typename T>
Tensor toInferDataAndResizeInPlace(std::vector<T> &&v, int64_t targetLength, T val) {
    int64_t shape[2] = {1, targetLength};
    if (targetLength > v.size()) {
        v.resize(targetLength, val);
    }
    return Tensor::create(v.data(), targetLength, shape, 2);
}

template<typename T_Src, typename T_Dst>
Tensor toInferDataAsType(const std::vector<T_Src> &v) {
    Tensor t;
    t.data.resize(v.size() * sizeof(T_Dst));
    T_Dst *buf;
    t.getDataBuffer<T_Dst>(&buf);

    for (const auto &item : v) {
        *(buf++) = static_cast<T_Dst>(item);
    }

    t.shape = {1, static_cast<int64_t>(v.size())};
    if constexpr (std::is_same_v<T_Dst, float>) {
        t.type = Tensor::Float;
    } else if constexpr (std::is_same_v<T_Dst, int64_t>) {
        t.type = Tensor::Int64;
    } else if constexpr (std::is_same_v<T_Dst, bool>) {
        t.type = Tensor::Bool;
    }
    return t;
}

template<typename T_Src, typename T_Dst>
Tensor toInferDataAsTypeAndResize(const std::vector<T_Src> &v, int64_t targetLength, T_Dst val) {
    int64_t shape[2] = {1, targetLength};

    if (v.empty()) {
        std::vector<T_Dst> v2(targetLength, val);
        return Tensor::create(v2.data(), v2.size(), shape, 2);
    }

    const int64_t len = v.size();
    if (targetLength <= len) {
        std::vector<T_Dst> v2(v.begin(), v.begin() + targetLength);
        return Tensor::create(v2.data(), v2.size(), shape, 2);
    }

    std::vector<T_Dst> v2(v.begin(), v.end());
    v2.resize(targetLength, val);
    return Tensor::create(v2.data(), v2.size(), shape, 2);
}


Tensor parsePhonemeTokens(
        const Segment &dsSegment,
        const std::unordered_map<std::string, int64_t> &name2token) {

    std::vector<int64_t> tokens;
    tokens.reserve(dsSegment.phoneCount());
    for (const auto &word : dsSegment.words) {
        for (const auto &phone : word.phones) {
            // tokens
            std::string tokenWithLang =
                    (phone.language.empty() || phone.token == "SP" || phone.token == "AP") ?
                    phone.token :
                    (phone.language + '/' + phone.token);
            if (const auto it = name2token.find(tokenWithLang); it != name2token.end()) {
                // first try finding the phoneme with the language tag (lang/phoneme)
                tokens.push_back(it->second);
            } else if (const auto it2 = name2token.find(phone.token); it2 != name2token.end()) {
                // then try finding the phoneme without the language tag (phoneme)
                tokens.push_back(it2->second);
            } else {
                // TODO: error handling
                tokens.push_back(0);
            }
        }
    }
    return toInferDataInPlace(std::move(tokens));
}


Tensor parsePhonemeLanguages(
        const Segment &dsSegment,
        const std::unordered_map<std::string, int64_t> &languages) {

    std::vector<int64_t> lang;
    lang.reserve(dsSegment.phoneCount());
    for (const auto &word : dsSegment.words) {
        for (const auto &phone : word.phones) {
            // tokens
            if (const auto it = languages.find(phone.language); it != languages.end()) {
                lang.push_back(it->second);
            } else {
                // TODO: error handling
                lang.push_back(0);
            }
        }
    }
    return toInferDataInPlace(std::move(lang));
}

Tensor parsePhonemeDurations(
        const Segment &dsSegment,
        double frameLength) {
    auto phoneCount = dsSegment.phoneCount();

    std::vector<int64_t> durations;
    durations.reserve(phoneCount);

    double phoneDurSum = 0.0;

    for (const auto &word : dsSegment.words) {
        auto wordDuration = word.duration();

        bool phoneDurFlag = false;
        double phoneDurPrevStart = 0.0;

        for (size_t i = 0; i < word.phones.size(); ++i) {

            // durations
            {
                auto currPhoneStart = phoneDurSum +
                                      word.phones[i].start;
                auto nextPhoneStart = phoneDurSum +
                                      ((i == word.phones.size() - 1) ? wordDuration : word.phones[i + 1].start);
                int64_t currPhoneStartFrames = std::llround(currPhoneStart / frameLength);
                int64_t nextPhoneStartFrames = std::llround(nextPhoneStart / frameLength);
                durations.push_back(nextPhoneStartFrames - currPhoneStartFrames);
            }
        }
        phoneDurSum += wordDuration;
    }

    return toInferDataInPlace(std::move(durations));
}



InferMap acousticPreprocess(
        const std::unordered_map<std::string, int64_t> &name2token,
        const std::unordered_map<std::string, int64_t> &languages,
        const Segment &dsSegment,
        const DsConfig &dsConfig,
        double frameLength,
        double transpose,
        Status *status) {

    InferMap m;

    bool isMultiLang = !languages.empty();
    m["tokens"] = parsePhonemeTokens(dsSegment, name2token);
    if (isMultiLang) {
        m["languages"] = parsePhonemeLanguages(dsSegment, languages);
    }

    auto durations = parsePhonemeDurations(dsSegment, frameLength);

    const int64_t *buffer;
    const auto bufferSize = durations.getDataBuffer<int64_t>(&buffer);

    int64_t targetLength = std::accumulate(buffer, buffer + bufferSize, int64_t{0});
    m["durations"] = durations;

    bool hasPitch = false;
    if (auto it = dsSegment.parameters.find("pitch"); it != dsSegment.parameters.end()) {
        const auto &param = it->second;
        auto samples = param.sample_curve.resample(frameLength, targetLength);
        if (param.tag == "pitch") {
            std::transform(samples.begin(), samples.end(), samples.begin(), [transpose](double midiPitch) {
                constexpr double referenceFrequency = 440.0;
                constexpr double semitonesInOctave = 12.0;
                constexpr double midiPitchOffset = 69.0;
                return (referenceFrequency *
                        std::pow(2.0, (midiPitch - midiPitchOffset + transpose) / semitonesInOctave));
            });
            m["f0"] = toInferDataAsType<double, float>(samples);
            hasPitch = true;
        }
        //m[param.tag] = toInferDataAsType<double, float>(samples);
    }
    // velocity, gender, energy, breathiness
    auto tryAddParam = [frameLength, targetLength, &dsSegment, &m](const std::string &paramName) {
        if (auto it = dsSegment.parameters.find(paramName); it != dsSegment.parameters.end()) {
            const auto &param = it->second;
            if (param.tag == paramName) {
                auto samples = param.sample_curve.resample(frameLength, targetLength);
                m[paramName] = toInferDataAsType<double, float>(samples);
                return true;
            }
        }
        return false;
    };

    auto tensor0 = toInferDataAsType<float, float>(std::vector<float>(targetLength, 0.0f));
    auto tensor1 = toInferDataAsType<float, float>(std::vector<float>(targetLength, 1.0f));
    if ((dsConfig.features & kfParamGender) && !tryAddParam("gender")) {
        m["gender"] = tensor0;
    }
    if ((dsConfig.features & kfParamVelocity) && !tryAddParam("velocity")) {
        m["velocity"] = tensor1;
    }
    std::vector<std::string> missingParameters;
    if ((dsConfig.features & kfParamBreathiness) && !tryAddParam("breathiness")) {
        missingParameters.emplace_back("breathiness");
    }
    if ((dsConfig.features & kfParamTension) && !tryAddParam("tension")) {
        missingParameters.emplace_back("tension");
    }
    if ((dsConfig.features & kfParamVoicing) && !tryAddParam("voicing")) {
        missingParameters.emplace_back("voicing");
    }
    if ((dsConfig.features & kfParamEnergy) && !tryAddParam("energy")) {
        missingParameters.emplace_back("energy");
    }

    if (!missingParameters.empty()) {
        std::string errMsg = "The acoustic model expects";
        for (const auto &missingParameter : missingParameters) {
            errMsg += " \"" + missingParameter + '\"';
        }
        errMsg += " but such parameters are not provided.";
        putStatus(status, Status_InferError, std::move(errMsg));
        return {};
    }
    // DONE: static spk_mix
    // TODO: curve spk_mix

    if (!dsConfig.speakers.empty()) {
        // Required to choose a speaker.
        // {1, N, 256}
        auto spkMix = getSpkMix(dsConfig.spkEmb, dsConfig.speakers, dsSegment.speakers, frameLength, targetLength);
        std::array<int64_t, 3> shape = {int64_t{1}, targetLength, static_cast<int64_t>(SPK_EMBED_SIZE)};
        m["spk_embed"] = Tensor::create(spkMix.data(), spkMix.size(), shape.data(), shape.size());
    }

    return m;
}

InferMap linguisticPreprocess(
        const std::unordered_map<std::string, int64_t> &name2token,
        const std::unordered_map<std::string, int64_t> &languages,
        const Segment &dsSegment,
        double frameLength,
        bool predictDur,
        Status *status) {
    InferMap m;

    bool isMultiLang = !languages.empty();
    m["tokens"] = parsePhonemeTokens(dsSegment, name2token);
    if (isMultiLang) {
        m["languages"] = parsePhonemeLanguages(dsSegment, languages);
    }

    if (predictDur) {
        std::vector<int64_t> wordDiv(dsSegment.words.size());
        std::transform(dsSegment.words.begin(), dsSegment.words.end(), wordDiv.begin(),
                       [](const Word &word) { return word.phones.size(); });

        m["word_div"] = toInferDataInPlace(std::move(wordDiv));

        std::vector<int64_t> wordDurFrames;
        wordDurFrames.reserve(dsSegment.words.size());

        int64_t wordDurSumPrevFrames = 0;
        double wordDurSumCurr = 0.0;
        for (const auto &word: dsSegment.words) {
            wordDurSumCurr += word.duration();
            int64_t wordDurSumCurrFrames = std::llround(wordDurSumCurr / frameLength);
            wordDurFrames.push_back(wordDurSumCurrFrames - wordDurSumPrevFrames);
            wordDurSumPrevFrames = wordDurSumCurrFrames;
        }
        m["word_dur"] = toInferDataInPlace(std::move(wordDurFrames));
    } else {
        m["ph_dur"] = parsePhonemeDurations(dsSegment, frameLength);
    }
    return m;
}

InferMap durPreprocess(
        const Segment &dsSegment,
        const DsDurConfig &dsDurConfig,
        Status *status) {
    InferMap m;
    auto phoneCount = dsSegment.phoneCount();
    std::vector<int64_t> phMidi;
    phMidi.reserve(phoneCount);

    constexpr int64_t restMidi = -127;

    bool nonRestOccurred = false;
    size_t restCountAtBeginning = 0;
    int64_t fillMidiForBeginning = 0;
    int64_t lastMidi = restMidi;


    for (const auto &word : dsSegment.words) {
        if (word.notes.empty()) {
            // TODO: error handling
            continue;
        }

        if (word.notes.size() == 1) {
            const auto &note = word.notes[0];
            if (!nonRestOccurred) {
                if (!note.is_rest) {
                    fillMidiForBeginning = note.key;
                    nonRestOccurred = true;
                } else {
                    restCountAtBeginning += word.phones.size();
                }
            } else {
                if (!note.is_rest) {
                    lastMidi = note.key;
                }
#if 1
                else {
                    lastMidi = restMidi;
                }
#endif
            }
            for (const auto &phone: word.phones) {
                phMidi.push_back(lastMidi);
            }
        } else {
            std::vector<double> noteCumDur;
            noteCumDur.reserve(word.notes.size());
            double s = 0.0;
            for (const auto &note: word.notes) {
                s += note.duration;
                noteCumDur.push_back(s);
            }

            for (const auto &phone: word.phones) {
                size_t noteIndex = 0;
                while (noteIndex < noteCumDur.size()) {
                    if (phone.start > noteCumDur[noteIndex]) {
                        break;
                    }
                    ++noteIndex;
                }
                if (noteIndex >= word.notes.size()) {
                    noteIndex = word.notes.size() - 1;
                }
                const auto &note = word.notes[noteIndex];
                if (!nonRestOccurred) {
                    if (!note.is_rest) {
                        fillMidiForBeginning = note.key;
                        nonRestOccurred = true;
                    } else {
                        ++restCountAtBeginning;
                    }
                } else {
                    if (!note.is_rest) {
                        lastMidi = note.key;
                    }
#if 1
                    else {
                        lastMidi = restMidi;
                    }
#endif
                }
                phMidi.push_back(lastMidi);
            }
        }
    }
    if (restCountAtBeginning >= phMidi.size()) {
        restCountAtBeginning = phMidi.size();
    }
    for (size_t i = 0; i < restCountAtBeginning; ++i) {
        phMidi[i] = fillMidiForBeginning;
    }

#if 1
    fillRestMidiWithNearestInPlace(phMidi, restMidi);
#endif
    m["ph_midi"] = toInferDataInPlace(std::move(phMidi));

    if (!dsDurConfig.speakers.empty()) {
        // Required to choose a speaker.
        // {1, N, 256}

        // TODO: Currently dur model always uses static mix.
        //       Consider allowing dynamic mix, but the axis is in phonemes instead of frames,
        //       so processing `spk_embed` in dur model is different than that in pitch, variance and acoustic models.
        std::unordered_map<std::string, double> staticMixMap;
        staticMixMap.reserve(dsSegment.speakers.spk.size());
        for (const auto &[key, value] : dsSegment.speakers.spk) {
            staticMixMap[key] = value.samples.empty() ? 0 : value.samples[0];
        }
        auto spkMix = getSpkMix(dsDurConfig.spkEmb, dsDurConfig.speakers, SpeakerMixCurve::fromStaticMix(staticMixMap),
                                1, static_cast<int64_t>(phoneCount));
        std::array<int64_t, 3> shape = {int64_t{1}, static_cast<int64_t>(phoneCount), static_cast<int64_t>(SPK_EMBED_SIZE)};
        m["spk_embed"] = Tensor::create(spkMix.data(), spkMix.size(), shape.data(), shape.size());
    }

    return m;
}


InferMap pitchProcess(
        const Segment &dsSegment,
        const DsPitchConfig &dsPitchConfig,
        double frameLength,
        bool predictDur,
        Status *status) {
    InferMap m;

    size_t noteCount = dsSegment.noteCount();
    std::vector<float> noteMidi;
    std::vector<unsigned char> noteRest;
    std::vector<int64_t> noteDur;
    noteMidi.reserve(noteCount);
    noteRest.reserve(noteCount);
    noteDur.reserve(noteCount);

    constexpr float restMidi = -127.0f;

    double noteDurSum = 0.0;
    for (const auto &word: dsSegment.words) {
        for (const auto &note : word.notes) {
            noteRest.push_back(note.is_rest ? 1 : 0);
            if (note.is_rest) {
                noteMidi.push_back(restMidi);
            } else {
                noteMidi.push_back(static_cast<float>(note.key) + static_cast<float>(note.cents) / 100.0f);
            }
            int64_t noteDurPrevFrames = std::llround(noteDurSum / frameLength);
            noteDurSum += note.duration;
            int64_t noteDurCurrFrames = std::llround(noteDurSum / frameLength);
            noteDur.push_back(noteDurCurrFrames - noteDurPrevFrames);
        }
    }

    int64_t nFrames = std::accumulate(noteDur.begin(), noteDur.end(), int64_t{0}, std::plus<>());

#if 1
    fillRestMidiWithNearestInPlace(noteMidi, restMidi);
#endif
    m["note_midi"] = toInferDataInPlace(std::move(noteMidi));
    if (dsPitchConfig.features & kfParamNoteRest) {
        m["note_rest"] = toInferDataAsType<unsigned char, bool>(noteRest);
    }
    m["note_dur"] = toInferDataInPlace(std::move(noteDur));

    if (predictDur) {
        // The linguistic model inputs word_div and word_dur.
        // So ph_dur should be an input of pitch model, instead of binding from linguistic model input.
        m["ph_dur"] = parsePhonemeDurations(dsSegment, frameLength);
    }

    if (auto it = dsSegment.parameters.find("pitch"); it != dsSegment.parameters.end()) {
        const auto &pitch = it->second;
        m["pitch"] = toInferDataAsType<double, float>(pitch.sample_curve.resample(frameLength, nFrames));
        int64_t newRetakeStart = std::clamp(
                std::llround(static_cast<double>(pitch.retake_start) * pitch.sample_curve.timestep / frameLength),
                int64_t{0},
                nFrames);
        auto newRetakeEnd = std::clamp(
                std::llround(static_cast<double>(pitch.retake_end) * pitch.sample_curve.timestep / frameLength),
                int64_t{0},
                nFrames);
        std::vector<unsigned char> retake(nFrames, 0);
        std::fill(retake.begin() + newRetakeStart, retake.begin() + newRetakeEnd, int64_t{1});
        m["retake"] = toInferDataAsType<unsigned char, bool>(retake);
    } else {
        // TODO: error handling
    }

    if (dsPitchConfig.features & kfParamExpr) {
        if (auto it = dsSegment.parameters.find("expr"); it != dsSegment.parameters.end()) {
            const auto &expr = it->second;
            m["expr"] = toInferDataAsType<double, float>(expr.sample_curve.resample(frameLength, nFrames));
        } else {
            // TODO: warn user that expr is not specified and will use 1.
            m["expr"] = toInferDataInPlace(std::vector<float>(nFrames, 1.0f));
        }
    }

    if (!dsPitchConfig.speakers.empty()) {
        // Required to choose a speaker.
        // {1, N, 256}
        auto spkMix = getSpkMix(dsPitchConfig.spkEmb, dsPitchConfig.speakers, dsSegment.speakers, frameLength, nFrames);
        std::array<int64_t, 3> shape = {int64_t{1}, nFrames, static_cast<int64_t>(SPK_EMBED_SIZE)};
        m["spk_embed"] = Tensor::create(spkMix.data(), spkMix.size(), shape.data(), shape.size());
    }

    return m;
}

InferMap variancePreprocess(
        const Segment &dsSegment,
        const DsVarianceConfig &dsVarianceConfig,
        double frameLength,
        bool predictDur,
        Status *status) {
    InferMap m;
    // TODO
    double durSum = 0.0;
    for (const auto &word : dsSegment.words) {
        durSum += word.duration();
    }

    int64_t nFrames = std::llround(durSum / frameLength);

    if (auto it = dsSegment.parameters.find("pitch"); it != dsSegment.parameters.end()) {
        const auto &pitch = it->second;
        m["pitch"] = toInferDataAsType<double, float>(pitch.sample_curve.resample(frameLength, nFrames));
    } else {
        putStatus(status, Status_InferError, "Missing parameter \"pitch\" from segment");
        return {};
    }

    if (predictDur) {
        // The linguistic model inputs word_div and word_dur.
        // So ph_dur should be an input of variance model, instead of binding from linguistic model input.
        m["ph_dur"] = parsePhonemeDurations(dsSegment, frameLength);
    }

    std::vector<unsigned char> retake;

    std::vector<std::string> expectParamNames;

    if (dsVarianceConfig.features & kfParamEnergy) {
        expectParamNames.emplace_back("energy");
    }
    if (dsVarianceConfig.features & kfParamBreathiness) {
        expectParamNames.emplace_back("breathiness");
    }
    if (dsVarianceConfig.features & kfParamTension) {
        expectParamNames.emplace_back("tension");
    }
    if (dsVarianceConfig.features & kfParamVoicing) {
        expectParamNames.emplace_back("voicing");
    }

    if (expectParamNames.empty()) {
        putStatus(status, Status_InferError,
                  "According to the variance model config, it does not predict any parameters. Please check the config!");
        return {};
    } else {
        retake.resize(expectParamNames.size() * nFrames);
    }

    for (int64_t i = 0; i < expectParamNames.size(); ++i) {
        const auto &paramName = expectParamNames[i];
        if (auto it = dsSegment.parameters.find(paramName); it != dsSegment.parameters.end()) {
            const auto &p = it->second;
            m[paramName] = toInferDataAsType<double, float>(p.sample_curve.resample(frameLength, nFrames));
            int64_t newRetakeStart = std::clamp(
                    std::llround(static_cast<double>(p.retake_start) * p.sample_curve.timestep / frameLength),
                    int64_t{0},
                    nFrames);
            auto newRetakeEnd = std::clamp(
                    std::llround(static_cast<double>(p.retake_end) * p.sample_curve.timestep / frameLength),
                    int64_t{0},
                    nFrames);
            std::fill(retake.begin() + nFrames * i + newRetakeStart,
                      retake.begin() + nFrames * i + newRetakeEnd,
                      int64_t{1});
        } else {
            // TODO: error handling
            m[paramName] = toInferDataInPlace(std::vector<float>(nFrames));
            std::fill(retake.begin() + nFrames * i,
                      retake.begin() + nFrames * (i + 1),
                      int64_t{1});
        }
    }

    auto retakeTensor = toInferDataAsType<unsigned char, bool>(retake);
    auto numParams = static_cast<int64_t>(expectParamNames.size());
    retakeTensor.shape = {int64_t{1}, static_cast<int64_t>(retake.size()) / numParams, numParams};
    m["retake"] = std::move(retakeTensor);

    if (!dsVarianceConfig.speakers.empty()) {
        // Required to choose a speaker.
        // {1, N, 256}
        auto spkMix = getSpkMix(dsVarianceConfig.spkEmb, dsVarianceConfig.speakers, dsSegment.speakers, frameLength, nFrames);
        std::array<int64_t, 3> shape = {int64_t{1}, nFrames, static_cast<int64_t>(SPK_EMBED_SIZE)};
        m["spk_embed"] = Tensor::create(spkMix.data(), spkMix.size(), shape.data(), shape.size());
    }

    return m;
}

template<typename T>
void fillRestMidiWithNearestInPlace(std::vector<T> &src, T restMidi) {
    auto not_zero = [restMidi](T x) constexpr { return x != restMidi; };
    auto it = std::find(src.begin(), src.end(), restMidi);
    auto it_left = std::find_if(src.begin(), it, not_zero);
    auto it_right = std::find_if(it, src.end(), not_zero);

    if (it == src.end() || it_right == src.end()) {
        return;
    }

    // fill zero values at beginning
    if (it_left == it) {
        std::fill(src.begin(), it_right, *it_right);
        it = it_right;
    }

    // middle and end
    while (it != src.end() || it_right != src.end()) {
        auto it_prev = it;
        it = std::find(it_prev, src.end(), restMidi);
        it_left = it - 1;
        it_right = std::find_if(it, src.end(), not_zero);
        if (it_right == src.end()) {
            // end
            std::fill(it, it_right, *it_left);
            break;
        }
        // middle
        auto dist = std::distance(it_left, it_right);
        auto left_fills = dist / 2;
        auto right_fills = dist - left_fills - 1;
        std::fill(it, it + left_fills, *it_left);
        std::fill(it_right - right_fills, it_right, *it_right);
    }
}

template<typename T>
std::vector<T> fillRestMidiWithNearest(const std::vector<T> &src, T restMidi) {
    std::vector<T> dst(src.begin(), src.end());
    fillRestMidiWithNearestInPlace(dst, restMidi);
    return dst;
}

std::vector<float> getSpkMix(const SpeakerEmbed &spkEmb, const std::vector<std::string> &speakers, const SpeakerMixCurve &spkMix, double frameLength, int64_t targetLength) {
    // Required to choose a speaker.
    std::vector<float> spk_embed;
    int64_t spkEmbedArraySize = targetLength * SPK_EMBED_SIZE;
    spk_embed.resize(spkEmbedArraySize);
    if (spkMix.empty()) {
        // Use the first one by default.
        auto emb = spkEmb.getMixedEmb({{speakers[0], 1.0}});
        for (size_t i = 0; i < spkEmbedArraySize; ++i) {
            spk_embed[i] = emb[i % SPK_EMBED_SIZE];
        }
    } else {
        auto spkMixResampled = spkMix.resample(frameLength, targetLength);
        for (int64_t i = 0; i < targetLength; ++i) {
            std::unordered_map<std::string, double> mix;
            double mixSum = std::accumulate(spkMixResampled.spk.begin(), spkMixResampled.spk.end(), 0.0,
                                            [i](double value, const auto &speakerItem) {
                                                return value + speakerItem.second.samples[i];
                                            });
            if (mixSum == 0) {
                mixSum = 1;
            }
            int64_t speakerIndex = 0;
            for (const auto &speakerItem : spkMixResampled.spk) {
                // If SampleCurve::resample guarantees the size of returned array is at least `targetLength`,
                // subscripting will not go out of range here.
                mix[speakerItem.first] = speakerItem.second.samples[i] / mixSum;
                ++speakerIndex;
            }
            auto emb = spkEmb.getMixedEmb(mix);
            int64_t y = i * SPK_EMBED_SIZE;
            for (int64_t j = 0; j < SPK_EMBED_SIZE; ++j) {
                spk_embed[y + j] = emb[j];
            }
        }
    }
    return spk_embed;
}

bool isFileExtJson(const std::filesystem::path &path) {
    if (path.empty()) {
        return false;
    }
    std::string fileExt = path.extension().string();
    if (fileExt.size() != 5) {
        return false;
    }
    std::transform(fileExt.begin() + 1, fileExt.end(), fileExt.begin() + 1, [](char c) { return std::tolower(c); });
    return fileExt == ".json";
}

bool readPhonemesFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out) {
    std::string line;
    std::ifstream phonemesFile(path);

    int64_t token = 0;
    while (std::getline(phonemesFile, line)) {
        // handle CRLF line endings on Linux and macOS
        if (!line.empty() && line[line.size() - 1] == '\r')
            line.erase(line.size() - 1);

        out.emplace(line, token);
        ++token;
    }
    return true;
}

bool readMultiLangPhonemesFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out) {
    std::ifstream phonemesFile(path);
    nlohmann::json j;
    phonemesFile >> j;
    out = std::move(j);
    return true;
}

bool readLangIdFile(const std::filesystem::path &path, std::unordered_map<std::string, int64_t> &out) {
    std::ifstream languagesFile(path);
    nlohmann::json j;
    languagesFile >> j;
    out = std::move(j);
    return true;
}

DSONNXINFER_END_NAMESPACE
