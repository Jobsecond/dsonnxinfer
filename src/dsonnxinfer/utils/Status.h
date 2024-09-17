#ifndef DS_ONNX_INFER_STATUS_H
#define DS_ONNX_INFER_STATUS_H

#include <string>

#include <dsonnxinfer/dsonnxinfer_global.h>

DSONNXINFER_BEGIN_NAMESPACE

enum StatusCode {
    Status_Ok = 0,
    Status_GenericError,
    Status_ParseError,
    Status_SerializationError,
    Status_ModelLoadError,
    Status_InferError,
};

struct DSONNXINFER_EXPORT Status {
    StatusCode code = Status_Ok;
    std::string msg;

    bool isOk() const {
        return code == Status_Ok;
    }
};

void putStatus(Status *status, StatusCode code, const std::string &msg);
void putStatus(Status *status, StatusCode code, std::string &&msg);
void putStatusOk(Status *status = nullptr);

DSONNXINFER_END_NAMESPACE

#endif