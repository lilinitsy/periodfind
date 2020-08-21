#ifndef __PF_LS_H__
#define __PF_LS_H__

#include <cstddef>
#include <cstdint>
#include <vector>

class LombScargle {
   private:
   public:
    /**
     * Constructs a LombScargle object.
     */
    LombScargle();

    /**
     * Computes the Lomb-Scargle periodogram for the input light curve.
     *
     * Computes the Lomb-Scargle periodogram for the input light curve (times
     * and mags). The periodogram is calculated for each period and time
     * derivative, then output in a host array indexed first by period index,
     * then time derivative index (i.e. (period 0, td 0), (period 0, td 1), ...
     * (period 1, td 0), (period 1, td 1) etc.) which is returned.
     *
     * Arguments should all be device pointers.
     *
     * @param times light curve datapoint times
     * @param mags light curve datapoint magnitudes
     * @param length length of the input light curve
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     *
     * @return device array of periodograms
     */
    float* DeviceCalcLS(const float* times,
                        const float* mags,
                        const size_t length,
                        const float* periods,
                        const float* period_dts,
                        const size_t num_periods,
                        const size_t num_p_dts) const;

    /**
     * Computes the Lomb-Scargle periodogram for the input light curve.
     *
     * Computes the Lomb-Scargle periodogram for the input light curve (times
     * and mags). The periodogram is calculated for each period and time
     * derivative, then output in a host array indexed first by period index,
     * then time derivative index (i.e. (period 0, td 0), (period 0, td 1), ...
     * (period 1, td 0), (period 1, td 1) etc.) which is returned.
     *
     * Arguments should all be host pointers.
     *
     * @param times light curve datapoint times
     * @param mags light curve datapoint magnitudes
     * @param length length of the input light curve
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     *
     * @return host array of periodograms
     */
    float* CalcLS(const float* times,
                  const float* mags,
                  const size_t length,
                  const float* periods,
                  const float* period_dts,
                  const size_t num_periods,
                  const size_t num_p_dts) const;

    /**
     * Computes the Lomb-Scargle periodogram for all input light curves.
     *
     * Computes the Lomb-Scargle periodogram for each input light curve (times
     * and mags). The light curves should be stored consecutively in the input
     * times and magnitudes arrays, and should have lengths as specified (in
     * order) in the lengths vector. For each light curve, the periodogram is
     * calculated for each period and time derivative, indexed first by period
     * index then time derivative index (i.e. (period 0, td 0), (period 0, td
     * 1), ... (period 1, td 0), (period 1, td 1) etc.). The periodogram
     * arrays are stored consecutively and output as a single host array.
     *
     * Arguments should all be host pointers.
     *
     * @param times ordered vector of light curve times
     * @param mags orderded vector of light curve magnitudes
     * @param lengths ordered vector of light curve lengths
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     *
     * @return host array of periodograms
     */
    float* CalcLSBatched(const std::vector<float*>& times,
                         const std::vector<float*>& mags,
                         const std::vector<size_t>& lengths,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts) const;
};

#endif