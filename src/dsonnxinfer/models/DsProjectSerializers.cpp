#include "DsProjectSerializers_p.h"

#include <dsonnxinfer/DsProject.h>
#include <dsonnxinfer/SampleCurve.h>

DSONNXINFER_BEGIN_NAMESPACE

void to_json(nlohmann::json &j, const Phoneme &phoneme) {
    j = {
        {"token", phoneme.token},
        {"language", phoneme.language},
        {"start", phoneme.start},
    };
}

void from_json(const nlohmann::json &j, Phoneme &phoneme) {
    j.at("token").get_to(phoneme.token);
    if (auto it = j.find("language"); it != j.end()) {
        if (it->is_string()) {
            phoneme.language = it->get<std::string>();
        }
    }
    j.at("start").get_to(phoneme.start);
}

void to_json(nlohmann::json &j, const Note &note) {
    j = {
        {"key", note.key},
        {"cents", note.cents},
        {"duration", note.duration},
        {"glide", note.glide == Glide_Up ? "up" : note.glide == Glide_Down ? "down" : "none"},
        {"is_rest", note.is_rest},
    };
}

void from_json(const nlohmann::json &j, Note &note) {
    j.at("key").get_to(note.key);
    if (auto it = j.find("cents"); it != j.end()) {
        if (it->is_number()) {
            note.cents = it->get<int>();
        } else {
            note.cents = 0;
        }
    } else {
        note.cents = 0;
    }
    j.at("duration").get_to(note.duration);
    if (auto it = j.find("glide"); it != j.end()) {
        if (it->is_string()) {
            auto glide = it->get<std::string>();
            if (glide == "up") {
                note.glide = Glide_Up;
            } else if (glide == "down") {
                note.glide = Glide_Down;
            } else {
                note.glide = Glide_None;
            }
        } else {
            note.glide = Glide_None;
        }
    } else {
        note.glide = Glide_None;
    }
    j.at("is_rest").get_to(note.is_rest);
}

void to_json(nlohmann::json &j, const Word &word) {
    j = {
        {"phones", word.phones},
        {"notes", word.notes},
    };
}

void from_json(const nlohmann::json &j, Word &word) {
    j.at("phones").get_to(word.phones);
    j.at("notes").get_to(word.notes);
}

void to_json(nlohmann::json &j, const Parameter &parameter) {
    j = {
        {"tag", parameter.tag},
        {"interval", parameter.sample_curve.timestep},
        {"dynamic", true},
        {"values", parameter.sample_curve.samples},
        {"retake", {
            {"start", parameter.retake_start},
            {"end", parameter.retake_end},
        }},
    };
}

void from_json(const nlohmann::json &j, Parameter &parameter) {
    j.at("tag").get_to(parameter.tag);
    if (j["dynamic"]) {
        j.at("interval").get_to(parameter.sample_curve.timestep);
        j.at("values").get_to(parameter.sample_curve.samples);
    } else {
        parameter.sample_curve.samples.resize(1);
        j.at("value").get_to(parameter.sample_curve.samples[0]);
        if (auto it = j.find("interval"); it != j.end()) {
            if (it->is_number()) {
                it->get_to(parameter.sample_curve.timestep);
            }
        }
    }

    if (auto it = j.find("retake"); it != j.end()) {
        if (it->is_object()) {
            auto it_start = it->find("start");
            auto it_end = it->find("end");
            if (it_start != it->end()) {
                parameter.retake_start = *it_start;
            } else {
                parameter.retake_start = 0;
            }
            if (it_end != it->end()) {
                parameter.retake_end = *it_end;
            } else {
                parameter.retake_end = parameter.sample_curve.samples.size();
            }
        }
    } else {
        parameter.retake_start = 0;
        parameter.retake_end = parameter.sample_curve.samples.size();
    }
}

void to_json(nlohmann::json &j, const Segment &segment) {
    j = {
        {"offset", segment.offset},
        {"words", segment.words},
    };
    for (const auto &p : segment.parameters) {
        j["parameters"].push_back(p.second);
    }
    if (!segment.speakers.empty()) {
        j["speakers"] = segment.speakers;
    }
}

void from_json(const nlohmann::json &j, Segment &segment) {
    if (auto it = j.find("offset"); it != j.end()) {
        if (it->is_number()) {
            segment.offset = it->get<double>();
        } else {
            segment.offset = 0;
        }
    } else {
        segment.offset = 0;
    }

    j.at("words").get_to(segment.words);

    if (auto it = j.find("parameters"); it != j.end()) {
        if (it->is_array()) {
            for (const auto& j_parameter : *it) {
                auto p = j_parameter.get<Parameter>();
                auto tag = p.tag;
                segment.parameters.emplace(std::move(tag), std::move(p));
            }
        }
    }

    if (auto it = j.find("speakers"); it != j.end()) {
        if (it->is_array()) {
            it->get_to(segment.speakers);
        }
    }
}

void to_json(nlohmann::json &j, const SpeakerMixCurve &spk) {
    for (const auto &[name, sc] : spk.spk) {
        nlohmann::json j_item;
        if (sc.samples.size() == 1) {
            j_item = {
                    {"name", name},
                    {"dynamic", false},
                    {"value", sc.samples[0]},
            };
        } else {
            j_item = {
                    {"name", name},
                    {"dynamic", true},
                    {"interval", sc.timestep},
                    {"values", sc.samples},
            };
        }
        j.push_back(std::move(j_item));
    }
}

void from_json(const nlohmann::json &j, SpeakerMixCurve &spk) {
    if (!j.is_array()) {
        return;
    }
    for (const auto &j_item : j) {
        SampleCurve sc;
        if (j_item["dynamic"]) {
            j_item["values"].get_to(sc.samples);
            j_item["interval"].get_to(sc.timestep);
        } else {
            sc.samples.resize(1);
            j_item["value"].get_to(sc.samples[0]);
            if (auto it = j_item.find("interval"); it != j_item.end()) {
                if (it->is_number()) {
                    it->get_to(sc.timestep);
                }
            }
        }
        spk.spk.emplace(j_item["name"], std::move(sc));
    }
}

DSONNXINFER_END_NAMESPACE