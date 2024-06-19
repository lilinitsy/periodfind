// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#include "ls.h"

#include <algorithm>

#include <cstdio>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

const float TWO_PI = M_PI * 2.0;

//
// Simple LombScargle Function Definitions
//

LombScargle::LombScargle() {}

//
// CUDA Kernels
//

__global__ void LombScargleKernel(const float* __restrict__ times,
                                  const float* __restrict__ mags,
                                  const size_t length,
                                  const float* __restrict__ periods,
                                  const float* __restrict__ period_dts,
                                  const size_t num_periods,
                                  const size_t num_period_dts,
                                  const LombScargle params,
                                  float* periodogram) {
    const size_t thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= num_periods || thread_y >= num_period_dts) {
        return;
    }

    // Period and period time derivative
    const float period = periods[thread_x];
    const float period_dt = period_dts[thread_y];

    // Time derivative correction factor.
    const float pdt_corr = (period_dt / period) * 0.5f;

    float mag_cos = 0.0f;
    float mag_sin = 0.0f;
    float cos_cos = 0.0f;
    float cos_sin = 0.0f;

    float cos, sin, i_part;

#pragma unroll
    for (size_t idx = 0; idx < length; idx++) {
        float t = times[idx];
        float mag = mags[idx];

        float t_corr = t - pdt_corr * t * t;
        float folded = fabsf(modff(t_corr / period, &i_part));

        __sincosf(TWO_PI * folded, &sin, &cos);

        mag_cos += mag * cos;
        mag_sin += mag * sin;
        cos_cos += cos * cos;
        cos_sin += cos * sin;
    }

    float sin_sin = static_cast<float>(length) - cos_cos;

    float cos_tau, sin_tau;
    __sincosf(0.5f * atan2f(2.0f * cos_sin, cos_cos - sin_sin), &sin_tau,
              &cos_tau);

    float numerator_l = cos_tau * mag_cos + sin_tau * mag_sin;
    numerator_l *= numerator_l;

    float numerator_r = cos_tau * mag_sin - sin_tau * mag_cos;
    numerator_r *= numerator_r;

    float denominator_l = cos_tau * cos_tau * cos_cos
                          + 2 * cos_tau * sin_tau * cos_sin
                          + sin_tau * sin_tau * sin_sin;

    float denominator_r = cos_tau * cos_tau * sin_sin
                          - 2 * cos_tau * sin_tau * cos_sin
                          + sin_tau * sin_tau * cos_cos;

    periodogram[thread_x * num_period_dts + thread_y] =
        0.5 * ((numerator_l / denominator_l) + (numerator_r / denominator_r));
}

//
// Wrapper Functions
//

float* LombScargle::DeviceCalcLS(const float* times,
                                 const float* mags,
                                 const size_t length,
                                 const float* periods,
                                 const float* period_dts,
                                 const size_t num_periods,
                                 const size_t num_p_dts) const {
    float* periodogram;
    gpuErrchk(
        cudaMalloc(&periodogram, num_periods * num_p_dts * sizeof(float)));

    const size_t x_threads = 256;
    const size_t y_threads = 1;
    const size_t x_blocks = ((num_periods + x_threads - 1) / x_threads);
    const size_t y_blocks = ((num_p_dts + y_threads - 1) / y_threads);

    const dim3 block_dim = dim3(x_threads, y_threads);
    const dim3 grid_dim = dim3(x_blocks, y_blocks);

    LombScargleKernel<<<grid_dim, block_dim>>>(times, mags, length, periods,
                                               period_dts, num_periods,
                                               num_p_dts, *this, periodogram);

    return periodogram;
}

void LombScargle::CalcLS(float* times,
                         float* mags,
                         size_t length,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts,
                         float* per_out) const {
    CalcLSBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                  std::vector<size_t>{length}, periods, period_dts, num_periods,
                  num_p_dts, per_out);
}

float* LombScargle::CalcLS(float* times,
                           float* mags,
                           size_t length,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts) const {
    return CalcLSBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                         std::vector<size_t>{length}, periods, period_dts,
                         num_periods, num_p_dts);
}

void LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                const std::vector<float*>& mags,
                                const std::vector<size_t>& lengths,
                                const float* periods,
                                const float* period_dts,
                                const size_t num_periods,
                                const size_t num_p_dts,
                                float* per_out) const {
    // TODO: Look at ways of batching data transfer.

    // Size of one periodogram out array, and total periodogram output size.
    const size_t per_points = num_periods * num_p_dts;
    const size_t per_out_size = per_points * sizeof(float);
    const size_t per_points_doubled = 2 * per_points;
    const size_t num_streams = 3;

    // Buffer size (large enough for longest light curve)
    auto max_length = std::max_element(lengths.begin(), lengths.end());
    const size_t buffer_length = *max_length;
    const size_t buffer_bytes = sizeof(float) * buffer_length;
    const size_t buffer_length_doubled = 2 * buffer_length;

    // Trial information
    float* dev_periods;
    float* dev_period_dts;

    // Intermediate conditional entropy memory
    float* dev_per;

    // Device data
    float* dev_times_buffer;
    float* dev_mags_buffer;

    // Kernel launch information
    const size_t x_threads = 512;
    const size_t y_threads = 1;
    const size_t x_blocks = ((num_periods + x_threads - 1) / x_threads);
    const size_t y_blocks = ((num_p_dts + y_threads - 1) / y_threads);
    const dim3 block_dim = dim3(x_threads, y_threads);
    const dim3 grid_dim = dim3(x_blocks, y_blocks);

    // Create 3 cuda streams to pipeline async operations; a loop would be
    // cleaner, but slowed execution
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Perform all allocations
    gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));
    gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&dev_per, per_out_size * num_streams));
    gpuErrchk(cudaMalloc(&dev_times_buffer, buffer_bytes * num_streams));
    gpuErrchk(cudaMalloc(&dev_mags_buffer, buffer_bytes * num_streams));

    /*
    Zero conditional entropy output
    This strictly does not seem to be necessary since every entry in dev_per
    should be written to, but it's good to do at initialization to avoid
    garbage being accidentally set. It can be done per loop too, but it
    probably isn't necessary and introduces overhead */
    gpuErrchk(cudaMemset(dev_per, 0, per_out_size * num_streams));

    // Changing the code to use an inner loop increases runtime quite a bit
    // It might be that CUDA's async calls aren't optimized by the compiler when
    // doing that
#pragma unroll
    for (size_t i = 0; i < lengths.size(); i += num_streams) {
        // Copy light curve into device buffer
        const size_t i_plus_1 = i + 1;
        const size_t i_plus_2 = i + 2;
        const size_t curve_bytes_i = lengths[i] * sizeof(float);

        cudaMemcpyAsync(dev_times_buffer, times[i], curve_bytes_i,
                        cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_mags_buffer, mags[i], curve_bytes_i,
                        cudaMemcpyHostToDevice, stream1);

        LombScargleKernel<<<grid_dim, block_dim, 0, stream1>>>(
            dev_times_buffer, dev_mags_buffer, lengths[i], dev_periods,
            dev_period_dts, num_periods, num_p_dts, *this, dev_per);

        if (i_plus_1 < lengths.size()) {
            const size_t curve_bytes_next = lengths[i_plus_1] * sizeof(float);
            gpuErrchk(cudaMemcpyAsync(dev_times_buffer + buffer_length,
                                      times[i_plus_1], curve_bytes_next,
                                      cudaMemcpyHostToDevice, stream2));
            gpuErrchk(cudaMemcpyAsync(dev_mags_buffer + buffer_length,
                                      mags[i_plus_1], curve_bytes_next,
                                      cudaMemcpyHostToDevice, stream2));

            LombScargleKernel<<<grid_dim, block_dim, 0, stream2>>>(
                dev_times_buffer + buffer_length,
                dev_mags_buffer + buffer_length, lengths[i_plus_1], dev_periods,
                dev_period_dts, num_periods, num_p_dts, *this,
                dev_per + per_points);
        }

        if (i_plus_2 < lengths.size()) {
            const size_t curve_bytes_next = lengths[i_plus_2] * sizeof(float);
            gpuErrchk(cudaMemcpyAsync(dev_times_buffer + buffer_length_doubled,
                                      times[i_plus_2], curve_bytes_next,
                                      cudaMemcpyHostToDevice, stream3));
            gpuErrchk(cudaMemcpyAsync(dev_mags_buffer + buffer_length_doubled,
                                      mags[i_plus_2], curve_bytes_next,
                                      cudaMemcpyHostToDevice, stream3));

            LombScargleKernel<<<grid_dim, block_dim, 0, stream3>>>(
                dev_times_buffer + buffer_length_doubled,
                dev_mags_buffer + buffer_length_doubled, lengths[i_plus_2],
                dev_periods, dev_period_dts, num_periods, num_p_dts, *this,
                dev_per + per_points_doubled);
        }

        // Copy periodogram back to host
        gpuErrchk(cudaMemcpyAsync(&per_out[i * per_points], dev_per,
                                  per_out_size, cudaMemcpyDeviceToHost,
                                  stream1));

        if (i_plus_1 < lengths.size()) {
            gpuErrchk(cudaMemcpyAsync(&per_out[i_plus_1 * per_points],
                                      dev_per + per_points, per_out_size,
                                      cudaMemcpyDeviceToHost, stream2));
        }

        if (i_plus_2 < lengths.size()) {
            gpuErrchk(cudaMemcpyAsync(
                &per_out[i_plus_2 * per_points], dev_per + per_points_doubled,
                per_out_size, cudaMemcpyDeviceToHost, stream3));
        }
    }

    gpuErrchk(cudaStreamSynchronize(stream1));
    gpuErrchk(cudaStreamSynchronize(stream2));
    gpuErrchk(cudaStreamSynchronize(stream3));
    gpuErrchk(cudaStreamDestroy(stream1));
    gpuErrchk(cudaStreamDestroy(stream2));
    gpuErrchk(cudaStreamDestroy(stream3));

    // Free all of the GPU memory
    gpuErrchk(cudaFree(dev_periods));
    gpuErrchk(cudaFree(dev_period_dts));
    gpuErrchk(cudaFree(dev_per));
    gpuErrchk(cudaFree(dev_times_buffer));
    gpuErrchk(cudaFree(dev_mags_buffer));
}

float* LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                  const std::vector<float*>& mags,
                                  const std::vector<size_t>& lengths,
                                  const float* periods,
                                  const float* period_dts,
                                  const size_t num_periods,
                                  const size_t num_p_dts) const {
    // Size of one periodogram out array, and total periodogram output size.
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    size_t per_size_total = per_out_size * lengths.size();

    // Allocate the output CE array so we can copy to it.
    float* per_out = (float*)malloc(per_size_total);

    CalcLSBatched(times, mags, lengths, periods, period_dts, num_periods,
                  num_p_dts, per_out);

    return per_out;
}