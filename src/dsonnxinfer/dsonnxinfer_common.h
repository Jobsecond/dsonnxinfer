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

DSONNXINFER_END_NAMESPACE

#endif // DSONNXINFER_COMMON_H