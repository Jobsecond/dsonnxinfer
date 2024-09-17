
#ifndef DS_ONNX_INFER_SAMPLECURVE_H
#define DS_ONNX_INFER_SAMPLECURVE_H

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <string>

#include <dsonnxinfer/dsonnxinfer_global.h>

DSONNXINFER_BEGIN_NAMESPACE

struct SampleCurve {
    std::vector<double> samples;
    double timestep = 0.0;

    SampleCurve();
    SampleCurve(const std::vector<double> &samples, double timestep);
    SampleCurve(std::vector<double> &&samples, double timestep);
    SampleCurve(double fillValue, int64_t targetLength, double targetTimestep);

    /**
     * @brief Resamples curve to target time step and length using interpolation.
     *
     * @param targetTimestep  The target curve time step.
     * @param targetLength    The target length of sample points.
     * @return                The target curve samples
     *
     * This function resamples original curve to target time step and length. The original curve
     * will be interpolated, and then resized to target length. If the size of interpolated vector is
     * smaller than target length, it will be truncated; otherwise, it will be expanded using the last value.
     */
    std::vector<double> resample(double targetTimestep, int64_t targetLength) const;
};

// TODO: still figuring out the format of spk_mix
struct SpeakerMixCurve {
    std::unordered_map<std::string, SampleCurve> spk;

    SpeakerMixCurve resample(double targetTimestep, int64_t targetLength) const;
    static SpeakerMixCurve fromStaticMix(const std::unordered_map<std::string, double> &spk,
                                         int64_t targetLength = 1,
                                         double targetTimestep = 1.0);
    inline size_t size() const;
    inline bool empty() const;
};

bool SpeakerMixCurve::empty() const {
    return spk.empty();
}

size_t SpeakerMixCurve::size() const {
    return spk.size();
}

DSONNXINFER_END_NAMESPACE


#endif //DS_ONNX_INFER_SAMPLECURVE_H
