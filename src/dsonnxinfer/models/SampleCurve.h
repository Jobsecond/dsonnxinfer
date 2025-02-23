
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
     * @param fillLast        A boolean indicating whether to use the last value for filling
     *                        when expanding. If true, the vector is expanded by appending the
     *                        last element; if false, zeros are appended.
     * @return                The target curve samples
     *
     * This function resamples the original curve to the target time step and length.
     * The original curve will be interpolated, then resized to the target length.
     * If the interpolated vector's size is smaller than the target length, it is expanded by
     * appending the last value (or zeros if `fillLast` is false). If larger, it is truncated.
     */
    std::vector<double> resample(double targetTimestep, int64_t targetLength, bool fillLast = true) const;
};

// TODO: still figuring out the format of spk_mix
struct SpeakerMixCurve {
    std::unordered_map<std::string, SampleCurve> spk;

    SpeakerMixCurve resample(double targetTimestep, int64_t targetLength) const;
    void resampleInPlace(double targetTimestep, int64_t targetLength);
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
