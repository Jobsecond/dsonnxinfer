#ifndef DSONNXINFER_COMMON_H
#define DSONNXINFER_COMMON_H

#include <dsonnxinfer/dsonnxinfer_global.h>

DSONNXINFER_BEGIN_NAMESPACE

enum ExecutionProvider {
    EP_CPU = 1,
    EP_DirectML = 2,
    EP_CUDA = 3,
    EP_CoreML = 4,
};

enum DsLoggingLevel {
    LOGGING_LEVEL_OFF = 0,
    LOGGING_LEVEL_FATAL = 1,
    LOGGING_LEVEL_ERROR = 2,
    LOGGING_LEVEL_WARNING = 3,
    LOGGING_LEVEL_INFO = 4,
    LOGGING_LEVEL_DEBUG = 5,
};

using DsLoggingCallback = void (*)(
    int,            // level
    const char *,   // category
    const char *    // message
);

DSONNXINFER_END_NAMESPACE

#endif // DSONNXINFER_COMMON_H