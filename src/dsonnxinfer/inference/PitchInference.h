#ifndef DSONNXINFER_PITCHINFERENCE_H
#define DSONNXINFER_PITCHINFERENCE_H

#include <memory>
#include <dsonnxinfer/dsonnxinfer_global.h>
#include "IInference.h"

DSONNXINFER_BEGIN_NAMESPACE

class DSONNXINFER_EXPORT PitchInference : public IInference {
public:
    explicit PitchInference(DsPitchConfig &&dsPitchConfig);
    explicit PitchInference(const DsPitchConfig &dsPitchConfig);
    ~PitchInference() override;

    Status open() override;
    void close() override;

    float depth() const;
    int64_t steps() const;
    void setDepth(float depth);
    void setSteps(int64_t steps);

    //InferMap infer(const Segment &dsSegment, Status *status) override;
    bool runInPlace(Segment &dsSegment, Status *status);
    bool terminate() override;

protected:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

DSONNXINFER_END_NAMESPACE

#endif //DSONNXINFER_PITCHINFERENCE_H
