#ifndef DS_ONNX_INFER_DSPROJECTSERIALIZERS_H
#define DS_ONNX_INFER_DSPROJECTSERIALIZERS_H

#include <dsonnxinfer/dsonnxinfer_global.h>

#include <nlohmann/json.hpp>

DSONNXINFER_BEGIN_NAMESPACE

struct Phoneme;
struct Note;
struct Word;
struct Parameter;
struct Segment;
struct SpeakerMixCurve;

void to_json(nlohmann::json &j, const Phoneme &phoneme);
void from_json(const nlohmann::json &j, Phoneme &phoneme);

void to_json(nlohmann::json &j, const Note &note);
void from_json(const nlohmann::json &j, Note &note);

void to_json(nlohmann::json &j, const Word &word);
void from_json(const nlohmann::json &j, Word &word);

void to_json(nlohmann::json &j, const Parameter &parameter);
void from_json(const nlohmann::json &j, Parameter &parameter);

void to_json(nlohmann::json &j, const Segment &segment);
void from_json(const nlohmann::json &j, Segment &segment);

void to_json(nlohmann::json &j, const SpeakerMixCurve &spk);
void from_json(const nlohmann::json &j, SpeakerMixCurve &spk);

DSONNXINFER_END_NAMESPACE

#endif //DS_ONNX_INFER_DSPROJECTSERIALIZERS_H