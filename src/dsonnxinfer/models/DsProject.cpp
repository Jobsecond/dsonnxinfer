#include "DsProject.h"
#include "DsProjectSerializers_p.h"

DSONNXINFER_BEGIN_NAMESPACE

static inline Segment deserializeHelper(nlohmann::json &&j, Status *status = nullptr) {
    try {
        if (!j.is_object()) {
            putStatus(status, Status_ParseError, "Invalid input request format! The outer must be an object.");
            return {};
        }
        putStatusOk(status);
        return j.get<Segment>();
    } catch (const nlohmann::json::invalid_iterator &e) {
        putStatus(status, Status_ParseError, std::string("Invalid input request format! ") + e.what());
    } catch (const nlohmann::json::type_error &e) {
        putStatus(status, Status_ParseError, std::string("Invalid input request format! ") + e.what());
    } catch (const nlohmann::json::out_of_range &e) {
        putStatus(status, Status_ParseError, std::string("Invalid input request format! ") + e.what());
    } catch (const nlohmann::json::other_error &e) {
        putStatus(status, Status_ParseError, std::string("Invalid input request format! ") + e.what());
    }
    return {};
}

static inline nlohmann::json serializeHelper(const Segment &segment, Status *status) {
    putStatusOk(status);
    return segment;
}

Segment Segment::fromJson(const std::string &inputString, Status *status) {
    try {
        return deserializeHelper(nlohmann::json::parse(inputString), status);
    } catch (const nlohmann::json::parse_error &e) {
        putStatus(status, Status_ParseError, std::string("Error parsing json: ") + e.what());
    }

    return {};
}

Segment Segment::fromCbor(const std::string &inputString, Status *status) {
    try {
        return deserializeHelper(nlohmann::json::from_cbor(inputString), status);
    } catch (const nlohmann::json::parse_error &e) {
        putStatus(status, Status_ParseError, std::string("Error parsing cbor: ") + e.what());
    }

    return {};
}

std::string Segment::toJson(Status *status) const {
    try {
        return serializeHelper(*this, status).dump();
    } catch (const std::exception &e) {
        putStatus(status, Status_SerializationError, std::string("Error serializing to JSON: ") + e.what());
        return {};
    }
}

std::string Segment::toCbor(Status *status) const {
    try {
        std::string result;
        nlohmann::json::to_cbor(serializeHelper(*this, status), result);
        return result;
    } catch (const std::exception &e) {
        putStatus(status, Status_SerializationError, std::string("Error serializing to CBOR: ") + e.what());
        return {};
    }
}

size_t Segment::phoneCount() const {
    size_t phoneCount = 0;
    for (const auto &word : words) {
        phoneCount += word.phones.size();
    }
    return phoneCount;
}

size_t Segment::noteCount() const {
    size_t noteCount = 0;
    for (const auto &word : words) {
        noteCount += word.notes.size();
    }
    return noteCount;
}

double Word::duration() const {
    double totalDuration = 0.0;
    for (const auto &note : notes) {
        totalDuration += note.duration;
    }
    return totalDuration;
}

DSONNXINFER_END_NAMESPACE




