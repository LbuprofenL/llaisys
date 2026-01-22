#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include <cstddef>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             const size_t B, const size_t K, const size_t M) {

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < M; ++j) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float acc = llaisys::utils::cast<float>(bias[j]);
                for (size_t k = 0; k < K; ++k) {
                    acc += llaisys::utils::cast<float>(in[k + i * K]) * llaisys::utils::cast<float>(weight[k + j * K]);
                }
                out[i * M + j] = llaisys::utils::cast<T>(acc);
            } else {
                T acc = bias[j];
                for (size_t k = 0; k < K; ++k) {
                    acc += in[k + i * K] * weight[k + j * K];
                }
                out[i * M + j] = acc;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, const size_t B, const size_t K, const size_t M) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias), B, K, M);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias), B, K, M);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias), B, K, M);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
