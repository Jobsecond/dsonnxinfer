#ifndef DS_ONNX_INFER_DSPROJECT_H
#define DS_ONNX_INFER_DSPROJECT_H

#include <vector>
#include <string>
#include <map>

#include <dsonnxinfer/dsonnxinfer_global.h>
#include <dsonnxinfer/SampleCurve.h>
#include <dsonnxinfer/SpeakerEmbed.h>
#include <dsonnxinfer/Status.h>

DSONNXINFER_BEGIN_NAMESPACE

enum GlideType {
    Glide_None = 0,
    Glide_Up = 1,
    Glide_Down = 2,
};

struct DSONNXINFER_EXPORT Phoneme {
    std::string token;
    std::string language;
    double start = 0.0;
};

struct DSONNXINFER_EXPORT Note {
    int key = 0;
    int cents = 0;
    double duration = 0.0;
    GlideType glide = Glide_None;
    bool is_rest = false;
};

struct DSONNXINFER_EXPORT Word {
    std::vector<Phoneme> phones;
    std::vector<Note> notes;

    double duration() const;
};

struct DSONNXINFER_EXPORT Parameter {
    std::string tag;
    SampleCurve sample_curve;
    size_t retake_start = 0;
    size_t retake_end = 0;
};

struct DSONNXINFER_EXPORT Segment {
    double offset = 0.0;
    std::vector<Word> words;
    std::map<std::string, Parameter> parameters;
    SpeakerMixCurve speakers;

    size_t phoneCount() const;
    size_t noteCount() const;

    std::string toJson(Status *status = nullptr) const;
    std::string toCbor(Status *status = nullptr) const;

    static Segment fromJson(const std::string &inputString, Status *status = nullptr);
    static Segment fromCbor(const std::string &inputString, Status *status = nullptr);
};


DSONNXINFER_END_NAMESPACE

#endif