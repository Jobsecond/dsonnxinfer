#ifndef DS_ONNX_INFER_IINFERENCE_H
#define DS_ONNX_INFER_IINFERENCE_H


#include <dsonnxinfer/dsonnxinfer_global.h>
#include <dsonnxinfer/Status.h>
#include <dsonnxinfer/DsConfig.h>
#include <dsonnxinfer/DsProject.h>

DSONNXINFER_BEGIN_NAMESPACE

enum InferenceType {
    IT_Unknown = 0,
    IT_Acoustic,
    IT_Vocoder,
    IT_Duration,
    IT_Pitch,
    IT_MultiVariance,
};

class DSONNXINFER_EXPORT IInference {
public:
    IInference();
    virtual ~IInference();

public:
    virtual Status open() = 0;
    virtual void close() = 0;
    //virtual InferMap infer(const Segment &dsSegment, Status *status) = 0;

protected:
    InferenceType m_type;
};

DSONNXINFER_END_NAMESPACE

#endif // DS_ONNX_INFER_IINFERENCE_H