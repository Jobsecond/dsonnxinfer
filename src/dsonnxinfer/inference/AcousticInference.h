#ifndef DSONNXINFER_ACOUSTICINFERENCE_H
#define DSONNXINFER_ACOUSTICINFERENCE_H

#include <memory>
#include <dsonnxinfer/dsonnxinfer_global.h>
#include "IInference.h"

DSONNXINFER_BEGIN_NAMESPACE

class DSONNXINFER_EXPORT AcousticInference : public IInference {
public:
    AcousticInference(DsConfig &&dsConfig,
                      DsVocoderConfig &&dsVocoderConfig);
    AcousticInference(const DsConfig &dsConfig,
                      const DsVocoderConfig &dsVocoderConfig);
    ~AcousticInference() override;

    Status open() override;
    void close() override;

    float depth() const;
    int64_t steps() const;
    void setDepth(float depth);
    void setSteps(int64_t steps);

    //InferMap infer(const Segment &dsSegment, Status *status) override;
    bool runAndSaveAudio(const Segment &dsSegment, const std::filesystem::path &path, Status *status);

    bool terminate() override;

protected:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

DSONNXINFER_END_NAMESPACE

#endif //DSONNXINFER_ACOUSTICINFERENCE_H
