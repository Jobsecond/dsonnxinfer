#include "Status.h"

DSONNXINFER_BEGIN_NAMESPACE

void putStatus(Status *status, StatusCode code, std::string &&msg) {
    if (status) {
        status->code = code;
        status->msg = msg;
    }
}

void putStatus(Status *status, StatusCode code, const std::string &msg) {
    if (status) {
        status->code = code;
        status->msg = msg;
    }
}

void putStatusOk(Status *status) {
    if (status) {
        status->code = Status_Ok;
        if (!status->msg.empty()) {
            status->msg = {};
        }
    }
}
DSONNXINFER_END_NAMESPACE