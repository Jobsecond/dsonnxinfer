#ifndef DSONNXINFER_DURATIONINFERENCE_H
#define DSONNXINFER_DURATIONINFERENCE_H

#include <memory>
#include <dsonnxinfer/dsonnxinfer_global.h>
#include "IInference.h"

DSONNXINFER_BEGIN_NAMESPACE

class DSONNXINFER_EXPORT DurationInference : public IInference {
public:
    explicit DurationInference(DsDurConfig &&dsDurConfig);
    explicit DurationInference(const DsDurConfig &dsDurConfig);
    ~DurationInference() override;

    Status open() override;
    void close() override;

    //InferMap infer(const Segment &dsSegment, Status *status) override;
    bool runInPlace(Segment &dsSegment, Status *status);

protected:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

DSONNXINFER_END_NAMESPACE

#endif //DSONNXINFER_DURATIONINFERENCE_H
