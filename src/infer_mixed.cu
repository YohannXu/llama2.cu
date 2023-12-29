#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_fp16.h>
#include <sentencepiece_processor.h>


void encode(sentencepiece::SentencePieceProcessor &processor, const std::string &text, std::vector<int> *ids, bool bos, bool eos) {
    processor.Encode(text, ids);
    if (bos) {
        ids->insert(ids->begin(), processor.bos_id());
    }
    if (eos) {
        ids->push_back(processor.eos_id());
    }
}


void decode(sentencepiece::SentencePieceProcessor &processor, const std::vector<int> ids, std::string *prompt) {
    processor.Decode(ids, prompt);
}


__device__ void clip(float* p_src, float min_val, float max_val) {
    if (p_src[0] < min_val) {
        p_src[0] = min_val;
    } else if (p_src[0] > max_val) {
        p_src[0] = max_val;
    }
}


__device__ void clip(half* p_src, half min_val, half max_val) {
    if (p_src[0] < min_val) {
        p_src[0] = min_val;
    } else if (p_src[0] > max_val) {
        p_src[0] = max_val;
    }
}


void get_freqs(char* freq_cos, char* freq_sin, int max_seq_len, int head_dim) {
    for (int i = 0; i < max_seq_len * 2; i++) {
        for (int j = 0; j < head_dim / 2; j++) {
            float cos = roundf(cosf(1.0 / powf(10000.0, 2 * float(j) / head_dim) * i) * 127);
            float sin = roundf(sinf(1.0 / powf(10000.0, 2 * float(j) / head_dim) * i) * 127);

            if (cos < -127.0) {
                cos = -127.0;
            } else if (cos > 127.0) {
                cos = 127.0;
            }
            freq_cos[i * head_dim / 2 + j] = char(cos);

            if (sin < -127.0) {
                sin = -127.0;
            } else if (sin > 127.0) {
                sin = 127.0;
            }
            freq_sin[i * head_dim / 2 + j] = char(sin);
        }
    }
}


void get_mask(char* mask, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j <= i) {
                mask[i * seq_len + j] = 1;
            } else {
                mask[i * seq_len + j] = 0;
            }
        }
    }
}


__global__ void Copy(char* src, char* dst, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    dst[idx] = src[idx];
}


__global__ void Copy(half* src, half* dst, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    dst[idx] = src[idx];
}


__global__ void RMSNormPowerMean(char* data, half* x_scale, half* dst, int seq_len, int dim) {
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = bs_idx * seq_len * dim + seq_idx * dim;
    int mean_idx = bs_idx * seq_len + seq_idx;

    float sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += powf(__half2float(half(data[idx + i]) * x_scale[0]), 2);
    }
    float mean = sum / dim;

    dst[mean_idx] = half(mean);
}


__global__ void RMSNormPowerMean(half* data, half* dst, int seq_len, int dim) {
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = bs_idx * seq_len * dim + seq_idx * dim;
    int mean_idx = bs_idx * seq_len + seq_idx;

    float sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += powf(data[idx + i], 2);
    }
    float mean = sum / dim;

    dst[mean_idx] = half(mean);
}


__global__ void RMSNormRsqrtMul(half* mean, char* data, char* w_norm, half* x_scale, half* w_scale, half* out_scale, float norm_eps, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    int mean_idx = bs_idx * seq_len + seq_idx;

    float rsqrt = rsqrtf(__half2float(mean[mean_idx]) + norm_eps);
    float x = __half2float(half(data[idx]) * x_scale[0]) * rsqrt;
    half res = roundf(__float2half(x) * half(w_norm[dim_idx]) * w_scale[0] / out_scale[0]);
    clip(&res, __float2half(-127.0), __float2half(127.0));
    data[idx] = char(res);
}


__global__ void RMSNormRsqrtMul(half* mean, half* data, char* dst, char* w_norm, half* w_scale, half* out_scale, float norm_eps, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    int mean_idx = bs_idx * seq_len + seq_idx;

    float rsqrt = rsqrtf(__half2float(mean[mean_idx]) + norm_eps);
    float x = __half2float(data[idx]) * rsqrt;
    half res = roundf(__float2half(x) * half(w_norm[dim_idx]) * w_scale[0] / out_scale[0]);
    clip(&res, __float2half(-127.0), __float2half(127.0));
    dst[idx] = char(res);
}


__global__ void Linear(char* data, char* weights, char* dst, half* x_scale, half* w_scale, half* out_scale, int seq_len, int dim_in, int dim_out) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim_out + seq_idx * dim_out + dim_idx;
    int offset = bs_idx * seq_len * dim_in + seq_idx * dim_in;

    int32_t sum = 0;
    for (int i = 0; i < dim_in; i++) {
        sum += data[offset + i] * weights[i * dim_out + dim_idx];
    }
    float res = roundf(float(sum) * __half2float(x_scale[0] * w_scale[0] / out_scale[0]));
    clip(&res, -127.0, 127.0);
    dst[idx] = char(res);
}


__global__ void Linear(char* data, char* weight, half* dst, half* x_scale, half* w_scale, int seq_len, int dim_in, int dim_out) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim_out + seq_idx * dim_out + dim_idx;
    int offset = bs_idx * seq_len * dim_in + seq_idx * dim_in;

    int32_t sum = 0;
    for (int i = 0; i < dim_in; i++) {
        sum += data[offset + i] * weight[i * dim_out + dim_idx];
    }
    float res = roundf(float(sum) * __half2float(x_scale[0]) * __half2float(w_scale[0]));
    dst[idx] = half(res);
}


__global__ void Linear(half* data, char* weight, half* dst, half* w_scale, int seq_len, int dim_in, int dim_out) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim_out + seq_idx * dim_out + dim_idx;
    int offset = bs_idx * seq_len * dim_in + seq_idx * dim_in;

    half sum = 0;
    for (int i = 0; i < dim_in; i++) {
        sum += data[offset + i] * (half(weight[i * dim_out + dim_idx]) * w_scale[0]);
    }
    dst[idx] = sum;
}


__global__ void RotaryEmb(char* data, char* dst, char* freq_cos, char* freq_sin, half* x_scale, half* out_scale, int seq_len, int dim, int n_heads, int head_dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    int freq_dim_idx = (dim_idx / 2) % (head_dim / 2);
    int freq_idx = seq_idx * head_dim / 2 + freq_dim_idx;

    half res;
    half cos_val = half(freq_cos[freq_idx]) / half(127);
    half sin_val = half(freq_sin[freq_idx]) / half(127);
    if (dim_idx % 2 == 0) {
        res = half(data[idx]) * x_scale[0] * cos_val - half(data[idx + 1]) * x_scale[0] * sin_val;
    } else {
        res = half(data[idx - 1]) * x_scale[0] * sin_val + half(data[idx]) * x_scale[0] * cos_val;
    }
    res = roundf(res / out_scale[0]);

    clip(&res, __float2half(-127.0), __float2half(127.0));
    dst[idx] = char(res);
}


__global__ void Cat(char* cache, char* data, char* dst, int seq_len, int total_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * total_len * dim + seq_idx * dim + dim_idx;

    if (seq_idx < total_len - seq_len) {
        dst[idx] = cache[bs_idx * (total_len - seq_len) * dim + seq_idx * dim + dim_idx];
    } else {
        dst[idx] = data[bs_idx * seq_len * dim + (seq_idx - total_len + seq_len) * dim + dim_idx];
    }
}


__global__ void TransposeSeqHeads(char* src, char* dst, int seq_len, int dim, int n_heads, int head_dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int src_idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    int n_heads_idx = dim_idx / head_dim;
    int head_dim_idx = dim_idx % head_dim;
    int dst_idx = bs_idx * seq_len * dim + n_heads_idx * seq_len * head_dim + seq_idx * head_dim + head_dim_idx;

    dst[dst_idx] = src[src_idx];
}


__global__ void TransposeSeqHeadDim(char* src, char* dst, int seq_len, int dim, int n_heads, int head_dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int n_heads_idx = dim_idx / head_dim;
    int head_dim_idx = dim_idx % head_dim;
    int src_idx = bs_idx * seq_len * dim + n_heads_idx * seq_len * head_dim + seq_idx * head_dim + head_dim_idx;
    int dst_idx = bs_idx * seq_len * dim + n_heads_idx * seq_len * head_dim + head_dim_idx * seq_len + seq_idx;

    dst[dst_idx] = src[src_idx];
}


__global__ void MatMul(char* x, char* y, char* dst, half* x_scale, half* y_scale, half* out_scale, int seq_len, int total_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * total_len + seq_idx * total_len + dim_idx;
    int x_offset = bs_idx * seq_len * dim + seq_idx * dim;
    int y_offset = bs_idx * total_len * dim + dim_idx;

    half sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (half(x[x_offset + i]) * x_scale[0]) * (half(y[y_offset + i * total_len]) * y_scale[0]);
    }
    sum = roundf(sum / out_scale[0]);
    clip(&sum, __float2half(-127.0), __float2half(127.0));
    dst[idx] = char(sum);
}


__global__ void MatMulScore(u_char* x, char* y, char* dst, half* y_scale, half* out_scale, int seq_len, int dim, int total_len) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    int x_offset = bs_idx * seq_len * total_len + seq_idx * total_len;
    int y_offset = bs_idx * total_len * dim + dim_idx;

    half sum = 0;
    for (int i = 0; i < total_len; i++) {
        sum += (half(x[x_offset + i]) / __float2half(255)) * (half(y[y_offset + i * dim]) * y_scale[0]);
    }
    sum = roundf(sum / out_scale[0]);
    clip(&sum, __float2half(-127.0), __float2half(127.0));
    dst[idx] = char(sum);
}


__global__ void Softmax(char* logit, char* mask, u_char* score, half* logit_scale, half factor, int seq_len, int total_len) {
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = bs_idx * seq_len * total_len + seq_idx * total_len;

    half max_val = __float2half(-10000.0);
    for (int i = 0; i < total_len; i++) {
        if (half(logit[offset + i]) * logit_scale[0] * factor > max_val) {
            max_val = half(logit[offset + i]) * logit_scale[0] * factor;
        }
    }

    float sum = 0;
    if (seq_len > 1) {
        for (int i = 0; i < total_len; i++) {
            sum += expf(half(logit[offset + i]) * logit_scale[0] * factor - max_val) * mask[seq_idx * seq_len + i];
        }
    } else {
        for (int i = 0; i < total_len; i++) {
            sum += expf(half(logit[offset + i]) * logit_scale[0] * factor - max_val);
        }
    }

    float prob;
    if (seq_len > 1) {
        for (int i = 0; i < total_len; i++) {
            prob = roundf(expf(half(logit[offset + i]) * logit_scale[0] * factor - max_val) * mask[seq_idx * seq_len + i] / sum * 255);
            clip(&prob, 0.0, 255.0);
            score[offset + i] = u_char(prob);
        }
    } else {
        for (int i = 0; i < total_len; i++) {
            prob = roundf(expf(half(logit[offset + i]) * logit_scale[0] * factor - max_val) / sum * 255);
            clip(&prob, 0.0, 255.0);
            score[offset + i] = u_char(prob);
        }
    }
}


__global__ void TransposeHeadsSeq(char* src, char* dst, int seq_len, int head_dim, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int n_heads_idx = dim_idx / head_dim;
    int head_dim_idx = dim_idx % head_dim;
    int src_idx = bs_idx * seq_len * dim + n_heads_idx * seq_len * head_dim + seq_idx * head_dim + head_dim_idx;
    int dst_idx = bs_idx * seq_len * dim + seq_idx * dim + n_heads_idx * head_dim + head_dim_idx;

    dst[dst_idx] = src[src_idx];
}


__global__ void Add(char* x, char* shortcut, half* x_scale, half* shortcut_scale, half* out_scale, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    half sum = half(x[idx]) * x_scale[0] + half(shortcut[idx]) * shortcut_scale[0];
    sum = roundf(sum / out_scale[0]);
    clip(&sum, __float2half(-127.0), __float2half(127.0));
    x[idx] = char(sum);
}


__global__ void Add(half* x, char* shortcut, half* shortcut_scale, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    half sum = x[idx] + half(shortcut[idx]) * shortcut_scale[0];
    x[idx] = sum;
}


__global__ void Add(char* x, half* shortcut, half* dst, half* x_scale, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    half sum = half(x[idx]) * x_scale[0] + shortcut[idx];
    dst[idx] = sum;
}


__global__ void SiLU(char* data, half* x_scale, half* out_scale, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    half x = half(data[idx]) * x_scale[0];
    x = x / __float2half(1 + expf(-x));
    half res = roundf(x / out_scale[0]);
    clip(&res, __float2half(-127.0), __float2half(127.0));
    data[idx] = char(res);
}


__global__ void SiLUHalf(char* data, half* dst, half* x_scale, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    half x = half(data[idx]) * x_scale[0];
    x = x / __float2half(1 + expf(-x));
    dst[idx] = x;
}


__global__ void Mul(char* x1, char* x3, half* x1_scale, half* x3_scale, half* out_scale, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    half res = (half(x1[idx]) * x1_scale[0]) * (half(x3[idx]) * x3_scale[0]);
    res = roundf(res / out_scale[0]);

    clip(&res, __float2half(-127.0), __float2half(127.0));
    x1[idx] = char(res);
}


__global__ void Mul(half* x1, half* x3, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    x1[idx] = x1[idx] * x3[idx];
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weight> <prompt>" << std::endl;
        return 1;
    }

    std::string weight_path = argv[1];
    std::string prompt = "<INST>";
    for (int i = 2; i < argc; i++) {
        prompt += " ";
        prompt += argv[i];
    }
    prompt += " </INST>";

    sentencepiece::SentencePieceProcessor processor;
    const auto status = processor.Load("weights/llama-7b-chat/tokenizer.model");
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
    }

    std::vector<int> tokens;
    encode(processor, prompt, &tokens, true, false);

    int bs = 1;
    int seq_len = tokens.size();
    int dim = 4096;
    int vocab_size = 32000;
    int n_layers = 32;
    int n_heads = 32;
    int max_seq_len = 128;
    int head_dim = dim / n_heads;
    int ff_dim = 256 * ((dim * 4 * 2 / 3 + 256 - 1) / 256);

    int n_elements = bs * seq_len * dim;
    int n_bytes = n_elements * sizeof(char);

    int n_weight_elements = dim * dim;
    int n_weight_bytes = n_weight_elements * sizeof(char);

    int n_output_elements = bs * seq_len * vocab_size;
    int n_output_bytes = n_output_elements * sizeof(half);

    float norm_eps = 1e-6;
    half dim_sqrt_inv = __float2half(rsqrtf(head_dim));

    int total_len = 0;
    int cache_offset, w_offset, w_ff_offset;

    char* h_input = (char*)malloc(n_bytes);
    half* h_output = (half*)malloc(n_output_bytes);

    char* h_freq_cos = (char*)malloc(max_seq_len * head_dim * sizeof(char));
    char* h_freq_sin = (char*)malloc(max_seq_len * head_dim * sizeof(char));
    get_freqs(h_freq_cos, h_freq_sin, max_seq_len, head_dim);
    char* h_mask = (char*)malloc(seq_len * seq_len * sizeof(char));
    get_mask(h_mask, seq_len);

    char* h_w_embed = (char*)malloc(vocab_size * dim * sizeof(char));
    char* h_w_norm = (char*)malloc((n_layers * 2 + 1) * dim * sizeof(char));
    char* h_wq = (char*)malloc(n_layers * n_weight_bytes);
    char* h_wk = (char*)malloc(n_layers * n_weight_bytes);
    char* h_wv = (char*)malloc(n_layers * n_weight_bytes);
    char* h_wo = (char*)malloc(n_layers * n_weight_bytes);
    char* h_w_ff1 = (char*)malloc(n_layers * ff_dim * dim * sizeof(char));
    char* h_w_ff2 = (char*)malloc(n_layers * ff_dim * dim * sizeof(char));
    char* h_w_ff3 = (char*)malloc(n_layers * ff_dim * dim * sizeof(char));
    char* h_w_out = (char*)malloc(vocab_size * dim * sizeof(char));

    half* h_scale_norm_x = (half*)malloc(4 * sizeof(half));
    half* h_scale_norm_w = (half*)malloc((n_layers * 2 + 1) * sizeof(half));
    half* h_scale_norm_out = (half*)malloc((n_layers * 2 + 1) * sizeof(half));
    half* h_scale_wq = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_wk = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_wv = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_xq = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_xk = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_xv = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_rotary_q = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_rotary_k = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_matmul = (half*)malloc(n_layers * 2 * sizeof(half));
    half* h_scale_wo = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_xo = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_ff_w1 = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_ff_x1 = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_ff_w3 = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_ff_x3 = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_silu = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_mul = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_ff_w2 = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_ff_x2 = (half*)malloc(n_layers * sizeof(half));
    half* h_scale_w_out = (half*)malloc(sizeof(half));

    std::ifstream weights(weight_path, std::ios::binary);

    if (!weights.is_open()) {
        std::cout << "open weights file failed!" << std::endl;
        return 0;
    }

    weights.read((char*)h_w_embed, vocab_size * dim * sizeof(char));
    weights.read((char*)h_w_norm, (n_layers * 2 + 1) * dim * sizeof(char));
    weights.read((char*)h_wq, n_layers * n_weight_bytes);
    weights.read((char*)h_wk, n_layers * n_weight_bytes);
    weights.read((char*)h_wv, n_layers * n_weight_bytes);
    weights.read((char*)h_wo, n_layers * n_weight_bytes);
    weights.read((char*)h_w_ff1, n_layers * ff_dim * dim * sizeof(char));
    weights.read((char*)h_w_ff2, n_layers * ff_dim * dim * sizeof(char));
    weights.read((char*)h_w_ff3, n_layers * ff_dim * dim * sizeof(char));
    weights.read((char*)h_w_out, vocab_size * dim * sizeof(char));

    weights.read((char*)h_scale_norm_x, 4 * sizeof(half));
    weights.read((char*)h_scale_norm_w, (n_layers * 2 + 1) * sizeof(half));
    weights.read((char*)h_scale_norm_out, (n_layers * 2 + 1) * sizeof(half));
    weights.read((char*)h_scale_wq, n_layers * sizeof(half));
    weights.read((char*)h_scale_wk, n_layers * sizeof(half));
    weights.read((char*)h_scale_wv, n_layers * sizeof(half));
    weights.read((char*)h_scale_xq, n_layers * sizeof(half));
    weights.read((char*)h_scale_xk, n_layers * sizeof(half));
    weights.read((char*)h_scale_xv, n_layers * sizeof(half));
    weights.read((char*)h_scale_rotary_q, n_layers * sizeof(half));
    weights.read((char*)h_scale_rotary_k, n_layers * sizeof(half));
    weights.read((char*)h_scale_matmul, n_layers * 2 * sizeof(half));
    weights.read((char*)h_scale_wo, n_layers * sizeof(half));
    weights.read((char*)h_scale_xo, n_layers * sizeof(half));
    weights.read((char*)h_scale_ff_w1, n_layers * sizeof(half));
    weights.read((char*)h_scale_ff_x1, n_layers * sizeof(half));
    weights.read((char*)h_scale_ff_w3, n_layers * sizeof(half));
    weights.read((char*)h_scale_ff_x3, n_layers * sizeof(half));
    weights.read((char*)h_scale_silu, n_layers * sizeof(half));
    weights.read((char*)h_scale_mul, n_layers * sizeof(half));
    weights.read((char*)h_scale_ff_w2, n_layers * sizeof(half));
    weights.read((char*)h_scale_ff_x2, n_layers * sizeof(half));
    weights.read((char*)h_scale_w_out, sizeof(half));

    weights.close();

    char* d_data;
    cudaMalloc((void**)&d_data, n_bytes);
    half* d_output;
    cudaMalloc((void**)&d_output, n_output_bytes);

    char* d_freq_cos;
    cudaMalloc((void**)&d_freq_cos, seq_len * head_dim * sizeof(char));
    char* d_freq_sin;
    cudaMalloc((void**)&d_freq_sin, seq_len * head_dim * sizeof(char));
    char* d_mask;
    cudaMalloc((void**)&d_mask, seq_len * seq_len * sizeof(char));
    cudaMemcpy(d_mask, h_mask, seq_len * seq_len * sizeof(char), cudaMemcpyHostToDevice);

    char* d_w_norm;
    cudaMalloc((void**)&d_w_norm, (n_layers * 2 + 1) * dim * sizeof(char));
    cudaMemcpy(d_w_norm, h_w_norm, (n_layers * 2 + 1) * dim * sizeof(char), cudaMemcpyHostToDevice);
    char* d_wq;
    cudaMalloc((void**)&d_wq, n_layers * n_weight_bytes);
    cudaMemcpy(d_wq, h_wq, n_layers * n_weight_bytes, cudaMemcpyHostToDevice);
    char* d_wk;
    cudaMalloc((void**)&d_wk, n_layers * n_weight_bytes);
    cudaMemcpy(d_wk, h_wk, n_layers * n_weight_bytes, cudaMemcpyHostToDevice);
    char* d_wv;
    cudaMalloc((void**)&d_wv, n_layers * n_weight_bytes);
    cudaMemcpy(d_wv, h_wv, n_layers * n_weight_bytes, cudaMemcpyHostToDevice);
    char* d_wo;
    cudaMalloc((void**)&d_wo, n_layers * n_weight_bytes);
    cudaMemcpy(d_wo, h_wo, n_layers * n_weight_bytes, cudaMemcpyHostToDevice);
    char* d_w_ff1;
    cudaMalloc((void**)&d_w_ff1, n_layers * ff_dim * dim * sizeof(char));
    cudaMemcpy(d_w_ff1, h_w_ff1, n_layers * ff_dim * dim * sizeof(char), cudaMemcpyHostToDevice);
    char* d_w_ff2;
    cudaMalloc((void**)&d_w_ff2, n_layers * ff_dim * dim * sizeof(char));
    cudaMemcpy(d_w_ff2, h_w_ff2, n_layers * ff_dim * dim * sizeof(char), cudaMemcpyHostToDevice);
    char* d_w_ff3;
    cudaMalloc((void**)&d_w_ff3, n_layers * ff_dim * dim * sizeof(char));
    cudaMemcpy(d_w_ff3, h_w_ff3, n_layers * ff_dim * dim * sizeof(char), cudaMemcpyHostToDevice);
    char* d_w_out;
    cudaMalloc((void**)&d_w_out, vocab_size * dim * sizeof(char));
    cudaMemcpy(d_w_out, h_w_out, vocab_size * dim * sizeof(char), cudaMemcpyHostToDevice);

    half* d_scale_norm_x;
    cudaMalloc((void**)&d_scale_norm_x, 4 * sizeof(half));
    cudaMemcpy(d_scale_norm_x, h_scale_norm_x, 4 * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_norm_w;
    cudaMalloc((void**)&d_scale_norm_w, (n_layers * 2 + 1) * sizeof(half));
    cudaMemcpy(d_scale_norm_w, h_scale_norm_w, (n_layers * 2 + 1) * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_norm_out;
    cudaMalloc((void**)&d_scale_norm_out, (n_layers * 2 + 1) * sizeof(half));
    cudaMemcpy(d_scale_norm_out, h_scale_norm_out, (n_layers * 2 + 1) * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_wq;
    cudaMalloc((void**)&d_scale_wq, n_layers * sizeof(half));
    cudaMemcpy(d_scale_wq, h_scale_wq, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_wk;
    cudaMalloc((void**)&d_scale_wk, n_layers * sizeof(half));
    cudaMemcpy(d_scale_wk, h_scale_wk, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_wv;
    cudaMalloc((void**)&d_scale_wv, n_layers * sizeof(half));
    cudaMemcpy(d_scale_wv, h_scale_wv, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_xq;
    cudaMalloc((void**)&d_scale_xq, n_layers * sizeof(half));
    cudaMemcpy(d_scale_xq, h_scale_xq, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_xk;
    cudaMalloc((void**)&d_scale_xk, n_layers * sizeof(half));
    cudaMemcpy(d_scale_xk, h_scale_xk, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_xv;
    cudaMalloc((void**)&d_scale_xv, n_layers * sizeof(half));
    cudaMemcpy(d_scale_xv, h_scale_xv, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_rotary_q;
    cudaMalloc((void**)&d_scale_rotary_q, n_layers * sizeof(half));
    cudaMemcpy(d_scale_rotary_q, h_scale_rotary_q, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_rotary_k;
    cudaMalloc((void**)&d_scale_rotary_k, n_layers * sizeof(half));
    cudaMemcpy(d_scale_rotary_k, h_scale_rotary_k, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_matmul;
    cudaMalloc((void**)&d_scale_matmul, n_layers * 2 * sizeof(half));
    cudaMemcpy(d_scale_matmul, h_scale_matmul, n_layers * 2 * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_wo;
    cudaMalloc((void**)&d_scale_wo, n_layers * sizeof(half));
    cudaMemcpy(d_scale_wo, h_scale_wo, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_xo;
    cudaMalloc((void**)&d_scale_xo, n_layers * sizeof(half));
    cudaMemcpy(d_scale_xo, h_scale_xo, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_ff_w1;
    cudaMalloc((void**)&d_scale_ff_w1, n_layers * sizeof(half));
    cudaMemcpy(d_scale_ff_w1, h_scale_ff_w1, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_ff_x1;
    cudaMalloc((void**)&d_scale_ff_x1, n_layers * sizeof(half));
    cudaMemcpy(d_scale_ff_x1, h_scale_ff_x1, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_ff_w3;
    cudaMalloc((void**)&d_scale_ff_w3, n_layers * sizeof(half));
    cudaMemcpy(d_scale_ff_w3, h_scale_ff_w3, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_ff_x3;
    cudaMalloc((void**)&d_scale_ff_x3, n_layers * sizeof(half));
    cudaMemcpy(d_scale_ff_x3, h_scale_ff_x3, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_silu;
    cudaMalloc((void**)&d_scale_silu, n_layers * sizeof(half));
    cudaMemcpy(d_scale_silu, h_scale_silu, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_mul;
    cudaMalloc((void**)&d_scale_mul, n_layers * sizeof(half));
    cudaMemcpy(d_scale_mul, h_scale_mul, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_ff_w2;
    cudaMalloc((void**)&d_scale_ff_w2, n_layers * sizeof(half));
    cudaMemcpy(d_scale_ff_w2, h_scale_ff_w2, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_ff_x2;
    cudaMalloc((void**)&d_scale_ff_x2, n_layers * sizeof(half));
    cudaMemcpy(d_scale_ff_x2, h_scale_ff_x2, n_layers * sizeof(half), cudaMemcpyHostToDevice);
    half* d_scale_w_out;
    cudaMalloc((void**)&d_scale_w_out, sizeof(half));
    cudaMemcpy(d_scale_w_out, h_scale_w_out, sizeof(half), cudaMemcpyHostToDevice);

    half* d_half_data;
    cudaMalloc((void**)&d_half_data, n_elements * sizeof(half));
    char* d_shortcut;
    cudaMalloc((void**)&d_shortcut, bs * seq_len * dim * sizeof(char));
    half* d_half_shortcut;
    cudaMalloc((void**)&d_half_shortcut, bs * seq_len * dim * sizeof(half));
    half* d_mean;
    cudaMalloc((void**)&d_mean, bs * seq_len * sizeof(half));
    char* d_xq;
    cudaMalloc((void**)&d_xq, n_bytes);
    char* d_xk;
    cudaMalloc((void**)&d_xk, n_bytes);
    char* d_xv;
    cudaMalloc((void**)&d_xv, n_bytes);
    char* d_rotary_q;
    cudaMalloc((void**)&d_rotary_q, n_bytes);
    char* d_rotary_k;
    cudaMalloc((void**)&d_rotary_k, n_bytes);
    char* k_caches;
    cudaMalloc((void**)&k_caches, n_layers * bs * max_seq_len * dim * sizeof(char));
    char* k_cache_cat;
    cudaMalloc((void**)&k_cache_cat, bs * max_seq_len * dim * sizeof(char));
    char* v_caches;
    cudaMalloc((void**)&v_caches, n_layers * bs * max_seq_len * dim * sizeof(char));
    char* v_cache_cat;
    cudaMalloc((void**)&v_cache_cat, bs * max_seq_len * dim * sizeof(char));
    char* keys;
    cudaMalloc((void**)&keys, n_layers * bs * max_seq_len * dim * sizeof(char));
    char* d_logit;
    cudaMalloc((void**)&d_logit, bs * n_heads * seq_len * max_seq_len * sizeof(char));
    u_char* d_score;
    cudaMalloc((void**)&d_score, bs * n_heads * seq_len * max_seq_len * sizeof(u_char));
    char* d_ff_x1;
    cudaMalloc((void**)&d_ff_x1, bs * seq_len * ff_dim * sizeof(char));
    char* d_ff_x3;
    cudaMalloc((void**)&d_ff_x3, bs * seq_len * ff_dim * sizeof(char));
    half* d_half_ff_x1;
    cudaMalloc((void**)&d_half_ff_x1, bs * seq_len * ff_dim * sizeof(half));
    half* d_half_ff_x3;
    cudaMalloc((void**)&d_half_ff_x3, bs * seq_len * ff_dim * sizeof(half));

    half max_prob = 0;
    int next_token = 0;
    std::vector<int> results_tokens;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    while (true) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < dim; j++) {
                h_input[i * dim + j] = h_w_embed[tokens[i] * dim + j];
            }
        }
        cudaMemcpy(d_data, h_input, bs * seq_len * dim * sizeof(char), cudaMemcpyHostToDevice);

        cudaMemcpy(d_freq_cos, h_freq_cos + total_len * head_dim / 2, seq_len * head_dim / 2 * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_freq_sin, h_freq_sin + total_len * head_dim / 2, seq_len * head_dim / 2 * sizeof(char), cudaMemcpyHostToDevice);

        total_len += seq_len;

        dim3 dimGrid(dim / 32, seq_len, bs);
        dim3 dimBlock(32, 1, 1);

        dim3 dimCacheGrid(dim / 32, total_len, bs);

        dim3 meanBlock(bs, seq_len);

        dim3 scoreGrid(total_len, seq_len, bs);
        dim3 scoreBlock(1, 1, n_heads);

        dim3 softmaxBlock(bs * n_heads, seq_len);

        dim3 attnGrid(head_dim, seq_len, bs);
        dim3 attnBlock(1, 1, n_heads);

        dim3 ffGrid(ff_dim / 32, seq_len, bs);
        dim3 ffBlock(32, 1, 1);

        dim3 outGrid(vocab_size / 32, seq_len, bs);
        dim3 outBlock(32, 1, 1);

        for (int layer_id = 0; layer_id < n_layers; layer_id++) {
            w_offset = layer_id * dim * dim;
            cache_offset = layer_id * bs * max_seq_len * dim;
            w_ff_offset = layer_id * ff_dim * dim;

            if (layer_id < 2) {
                Copy<<<dimGrid, dimBlock>>>(d_data, d_shortcut, seq_len, dim);
                RMSNormPowerMean<<<1, meanBlock>>>(d_data, d_scale_norm_x + layer_id * 2, d_mean, seq_len, dim);
                RMSNormRsqrtMul<<<dimGrid, dimBlock>>>(d_mean, d_data, d_w_norm + layer_id * 2 * dim, d_scale_norm_x + layer_id * 2, d_scale_norm_w + layer_id * 2, d_scale_norm_out + layer_id * 2, norm_eps, seq_len, dim);
            } else {
                Copy<<<dimGrid, dimBlock>>>(d_half_data, d_half_shortcut, seq_len, dim);
                RMSNormPowerMean<<<1, meanBlock>>>(d_half_data, d_mean, seq_len, dim);
                RMSNormRsqrtMul<<<dimGrid, dimBlock>>>(d_mean, d_half_data, d_data, d_w_norm + layer_id * 2 * dim, d_scale_norm_w + layer_id * 2, d_scale_norm_out + layer_id * 2, norm_eps, seq_len, dim);
            }

            Linear<<<dimGrid, dimBlock>>>(d_data, d_wq + w_offset, d_xq, d_scale_norm_out + layer_id * 2, d_scale_wq + layer_id, d_scale_xq + layer_id, seq_len, dim, dim);
            Linear<<<dimGrid, dimBlock>>>(d_data, d_wk + w_offset, d_xk, d_scale_norm_out + layer_id * 2, d_scale_wk + layer_id, d_scale_xk + layer_id, seq_len, dim, dim);
            Linear<<<dimGrid, dimBlock>>>(d_data, d_wv + w_offset, d_xv, d_scale_norm_out + layer_id * 2, d_scale_wv + layer_id, d_scale_xv + layer_id, seq_len, dim, dim);

            RotaryEmb<<<dimGrid, dimBlock>>>(d_xq, d_rotary_q, d_freq_cos, d_freq_sin, d_scale_xq + layer_id, d_scale_rotary_q + layer_id, seq_len, dim, n_heads, head_dim);
            RotaryEmb<<<dimGrid, dimBlock>>>(d_xk, d_rotary_k, d_freq_cos, d_freq_sin, d_scale_xk + layer_id, d_scale_rotary_k + layer_id, seq_len, dim, n_heads, head_dim);

            Cat<<<dimCacheGrid, dimBlock>>>(k_caches + cache_offset, d_rotary_k, k_cache_cat, seq_len, total_len, dim);
            Cat<<<dimCacheGrid, dimBlock>>>(v_caches + cache_offset, d_xv, v_cache_cat, seq_len, total_len, dim);
            cudaMemcpy(k_caches + cache_offset, k_cache_cat, bs * total_len * dim * sizeof(char), cudaMemcpyDeviceToDevice);
            cudaMemcpy(v_caches + cache_offset, v_cache_cat, bs * total_len * dim * sizeof(char), cudaMemcpyDeviceToDevice);

            TransposeSeqHeads<<<dimGrid, dimBlock>>>(d_rotary_q, d_xq, seq_len, dim, n_heads, head_dim);
            TransposeSeqHeads<<<dimCacheGrid, dimBlock>>>(k_caches + cache_offset, k_cache_cat, total_len, dim, n_heads, head_dim);
            TransposeSeqHeads<<<dimCacheGrid, dimBlock>>>(v_caches + cache_offset, v_cache_cat, total_len, dim, n_heads, head_dim);
            TransposeSeqHeadDim<<<dimCacheGrid, dimBlock>>>(k_cache_cat, keys, total_len, dim, n_heads, head_dim);

            MatMul<<<scoreGrid, scoreBlock>>>(d_xq, keys, d_logit, d_scale_rotary_q + layer_id, d_scale_rotary_k + layer_id, d_scale_matmul + layer_id * 2, seq_len, total_len, head_dim);
            Softmax<<<1, softmaxBlock>>>(d_logit, d_mask, d_score, d_scale_matmul + layer_id * 2, dim_sqrt_inv, seq_len, total_len);
            MatMulScore<<<attnGrid, attnBlock>>>(d_score, v_cache_cat, d_rotary_q, d_scale_xv + layer_id, d_scale_matmul + layer_id * 2 + 1, seq_len, head_dim, total_len);

            TransposeHeadsSeq<<<dimGrid, dimBlock>>>(d_rotary_q, d_rotary_k, seq_len, head_dim, dim);

            Linear<<<dimGrid, dimBlock>>>(d_rotary_k, d_wo + w_offset, d_data, d_scale_matmul + layer_id * 2 + 1, d_scale_wo + layer_id, d_scale_xo + layer_id, seq_len, dim, dim);

            if (layer_id < 2) {
                Add<<<dimGrid, dimBlock>>>(d_data, d_shortcut, d_scale_xo + layer_id, d_scale_norm_x + layer_id * 2, d_scale_norm_x + layer_id * 2 + 1, seq_len, dim);
            } else {
                Add<<<dimGrid, dimBlock>>>(d_data, d_half_shortcut, d_half_data, d_scale_xo + layer_id, seq_len, dim);
            }

            if (layer_id < 2) {
                Copy<<<dimGrid, dimBlock>>>(d_data, d_shortcut, seq_len, dim);
                RMSNormPowerMean<<<1, meanBlock>>>(d_data, d_scale_norm_x + layer_id * 2 + 1, d_mean, seq_len, dim);
                RMSNormRsqrtMul<<<dimGrid, dimBlock>>>(d_mean, d_data, d_w_norm + (layer_id * 2 + 1) * dim, d_scale_norm_x + layer_id * 2 + 1, d_scale_norm_w + layer_id * 2 + 1, d_scale_norm_out + layer_id * 2 + 1, norm_eps, seq_len, dim);
            } else {
                Copy<<<dimGrid, dimBlock>>>(d_half_data, d_half_shortcut, seq_len, dim);
                RMSNormPowerMean<<<1, meanBlock>>>(d_half_data, d_mean, seq_len, dim);
                RMSNormRsqrtMul<<<dimGrid, dimBlock>>>(d_mean, d_half_data, d_data, d_w_norm + (layer_id * 2 + 1) * dim, d_scale_norm_w + layer_id * 2 + 1, d_scale_norm_out + layer_id * 2 + 1, norm_eps, seq_len, dim);
            }

            Linear<<<ffGrid, ffBlock>>>(d_data, d_w_ff1 + w_ff_offset, d_ff_x1, d_scale_norm_out + layer_id * 2 + 1, d_scale_ff_w1 + layer_id, d_scale_ff_x1 + layer_id, seq_len, dim, ff_dim);
            if (layer_id == 1) {
                Linear<<<ffGrid, ffBlock>>>(d_data, d_w_ff3 + w_ff_offset, d_half_ff_x3, d_scale_norm_out + layer_id * 2 + 1, d_scale_ff_w3 + layer_id, seq_len, dim, ff_dim);
                SiLUHalf<<<ffGrid, ffBlock>>>(d_ff_x1, d_half_ff_x1, d_scale_ff_x1 + layer_id, seq_len, ff_dim);
                Mul<<<ffGrid, ffBlock>>>(d_half_ff_x1, d_half_ff_x3, seq_len, ff_dim);
                Linear<<<dimGrid, dimBlock>>>(d_half_ff_x1, d_w_ff2 + w_ff_offset, d_half_data, d_scale_ff_w2 + layer_id, seq_len, ff_dim, dim);
            } else {
                Linear<<<ffGrid, ffBlock>>>(d_data, d_w_ff3 + w_ff_offset, d_ff_x3, d_scale_norm_out + layer_id * 2 + 1, d_scale_ff_w3 + layer_id, d_scale_ff_x3 + layer_id, seq_len, dim, ff_dim);
                SiLU<<<ffGrid, ffBlock>>>(d_ff_x1, d_scale_ff_x1 + layer_id, d_scale_silu + layer_id, seq_len, ff_dim);
                Mul<<<ffGrid, ffBlock>>>(d_ff_x1, d_ff_x3, d_scale_silu + layer_id, d_scale_ff_x3 + layer_id, d_scale_mul + layer_id, seq_len, ff_dim);
                Linear<<<dimGrid, dimBlock>>>(d_ff_x1, d_w_ff2 + w_ff_offset, d_data, d_scale_mul + layer_id, d_scale_ff_w2 + layer_id, d_scale_ff_x2 + layer_id, seq_len, ff_dim, dim);
            }

            if (layer_id == 0) {
                Add<<<dimGrid, dimBlock>>>(d_data, d_shortcut, d_scale_ff_x2 + layer_id, d_scale_norm_x + layer_id * 2 + 1, d_scale_norm_x + (layer_id + 1) * 2, seq_len, dim);
            } else if (layer_id == 1) {
                Add<<<dimGrid, dimBlock>>>(d_half_data, d_shortcut, d_scale_norm_x + layer_id * 2 + 1, seq_len, dim);
            } else if (layer_id > 1) {
                Add<<<dimGrid, dimBlock>>>(d_data, d_half_shortcut, d_half_data, d_scale_ff_x2 + layer_id, seq_len, dim);
            }
        }

        RMSNormPowerMean<<<1, meanBlock>>>(d_half_data, d_mean, seq_len, dim);
        RMSNormRsqrtMul<<<dimGrid, dimBlock>>>(d_mean, d_half_data, d_data, d_w_norm + n_layers * 2 * dim, d_scale_norm_w + n_layers * 2, d_scale_norm_out + n_layers * 2, norm_eps, seq_len, dim);

        Linear<<<outGrid, outBlock>>>(d_data, d_w_out, d_output, d_scale_norm_out + n_layers * 2, d_scale_w_out, seq_len, dim, vocab_size);

        cudaMemcpy(h_output, d_output, bs * seq_len * vocab_size * sizeof(half), cudaMemcpyDeviceToHost);

        max_prob = -10000;
        for (int i = 0; i < vocab_size; i++) {
            if (h_output[(seq_len - 1) * vocab_size + i] > max_prob) {
                max_prob = h_output[(seq_len - 1) * vocab_size + i];
                next_token = i;
            }
        }

        results_tokens.push_back(next_token);

        if (next_token == processor.eos_id()) {
            break;
        } else if (total_len >= max_seq_len) {
            break;
        }

        tokens = {next_token};
        seq_len = 1;

    }

    decode(processor, results_tokens, &prompt);
    std::cout << prompt << std::endl;

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "inference time for " << total_len << " tokens : " << diff.count() << " s" << std::endl;
    std::cout << "average inference time for single token : " << diff.count() / total_len << " s" << std::endl;

    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_freq_cos);
    cudaFree(d_freq_sin);
    cudaFree(d_mask);

    cudaFree(d_w_norm);
    cudaFree(d_wq);
    cudaFree(d_wk);
    cudaFree(d_wv);
    cudaFree(d_wo);
    cudaFree(d_w_ff1);
    cudaFree(d_w_ff2);
    cudaFree(d_w_ff3);
    cudaFree(d_w_out);

    cudaFree(d_scale_norm_x);
    cudaFree(d_scale_norm_w);
    cudaFree(d_scale_norm_out);
    cudaFree(d_scale_wq);
    cudaFree(d_scale_wk);
    cudaFree(d_scale_wv);
    cudaFree(d_scale_xq);
    cudaFree(d_scale_xk);
    cudaFree(d_scale_xv);
    cudaFree(d_scale_rotary_q);
    cudaFree(d_scale_rotary_k);
    cudaFree(d_scale_matmul);
    cudaFree(d_scale_wo);
    cudaFree(d_scale_xo);
    cudaFree(d_scale_ff_w1);
    cudaFree(d_scale_ff_x1);
    cudaFree(d_scale_ff_w3);
    cudaFree(d_scale_ff_x3);
    cudaFree(d_scale_silu);
    cudaFree(d_scale_mul);
    cudaFree(d_scale_ff_w2);
    cudaFree(d_scale_ff_x2);
    cudaFree(d_scale_w_out);

    cudaFree(d_half_data);
    cudaFree(d_shortcut);
    cudaFree(d_half_shortcut);
    cudaFree(d_mean);
    cudaFree(d_xq);
    cudaFree(d_xk);
    cudaFree(d_xv);
    cudaFree(d_rotary_q);
    cudaFree(d_rotary_k);
    cudaFree(k_caches);
    cudaFree(k_cache_cat);
    cudaFree(v_caches);
    cudaFree(v_cache_cat);
    cudaFree(keys);
    cudaFree(d_logit);
    cudaFree(d_score);
    cudaFree(d_ff_x1);
    cudaFree(d_ff_x3);
    cudaFree(d_half_ff_x1);
    cudaFree(d_half_ff_x3);

    free(h_input);
    free(h_output);
    free(h_freq_cos);
    free(h_freq_sin);
    free(h_mask);

    free(h_w_embed);
    free(h_w_norm);
    free(h_wq);
    free(h_wk);
    free(h_wv);
    free(h_wo);
    free(h_w_ff1);
    free(h_w_ff2);
    free(h_w_ff3);
    free(h_w_out);

    free(h_scale_norm_x);
    free(h_scale_norm_w);
    free(h_scale_norm_out);
    free(h_scale_wq);
    free(h_scale_wk);
    free(h_scale_wv);
    free(h_scale_xq);
    free(h_scale_xk);
    free(h_scale_xv);
    free(h_scale_rotary_q);
    free(h_scale_rotary_k);
    free(h_scale_matmul);
    free(h_scale_wo);
    free(h_scale_xo);
    free(h_scale_ff_w1);
    free(h_scale_ff_x1);
    free(h_scale_ff_w3);
    free(h_scale_ff_x3);
    free(h_scale_silu);
    free(h_scale_mul);
    free(h_scale_ff_w2);
    free(h_scale_ff_x2);
    free(h_scale_w_out);

    return 0;
}