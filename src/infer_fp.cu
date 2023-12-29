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


void gen_freqs(int max_seq_len, int head_dim, half* h_freq_cos, half* h_freq_sin) {
    for (int i = 0; i < max_seq_len * 2; i++) {
        for (int j = 0; j < head_dim / 2; j++) {
            h_freq_cos[i * head_dim / 2 + j] = __float2half(cosf(1.0 / powf(10000.0, 2 * float(j) / head_dim) * i));
            h_freq_sin[i * head_dim / 2 + j] = __float2half(sinf(1.0 / powf(10000.0, 2 * float(j) / head_dim) * i));
        }
    }
}


__global__ void Half2Float(half* p_half, float* p_float, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    p_float[idx] = __half2float(p_half[idx]);
}


__global__ void Float2Half(float* p_float, half* p_half, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    p_half[idx] = __float2half(p_float[idx]);
}


__global__ void Copy(half* src, half* dst, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    dst[idx] = src[idx];
}


__global__ void Power(float* p_data, float* p_power_data, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    p_power_data[idx] = p_data[idx] * p_data[idx];
}


__global__ void Mean(float* p_data, float* p_mean, int seq_len, int dim) {
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    int idx = bs_idx * seq_len * dim + seq_idx * dim;
    int mean_idx = bs_idx * seq_len + seq_idx;
    for (int i = 0; i < dim; i++) {
        sum += p_data[idx + i];
    }
    p_mean[mean_idx] = sum / dim;
}


__global__ void AddMean(float* p_mean, float eps, int seq_len) {
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = bs_idx * seq_len + seq_idx;
    p_mean[idx] += eps;
}


__global__ void Rsqrt(float* p_mean, int seq_len) {
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = bs_idx * seq_len + seq_idx;
    p_mean[idx] = rsqrtf(p_mean[idx]);
}


__global__ void MulMeanRsqrt(float* p_data, float* p_mean, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    int mean_idx = bs_idx * seq_len + seq_idx;
    p_data[idx] *= p_mean[mean_idx];
}


__global__ void MulNormWeight(half* p_data, half* weight, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    p_data[idx] = __hmul(p_data[idx], weight[dim_idx]);
}


__global__ void MulVal(half* p_data, half* weight, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    p_data[idx] = __hmul(p_data[idx], weight[0]);
}


__global__ void Linear(half* p_src, half* weight, half* p_dst, int seq_len, int dim_in, int dim_out) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * dim_out + seq_idx * dim_out + dim_idx;
    int src_idx = bs_idx * seq_len * dim_in + seq_idx * dim_in;

    half sum = __float2half(0.0);
    for (int i = 0; i < dim_in; i++) {
        sum = __hadd(sum, __hmul(p_src[src_idx + i], weight[i * dim_out + dim_idx]));
    }
    p_dst[idx] = sum;
}


__global__ void RotaryEmb(half* src, half* dst, half* cos, half* sin, int seq_len, int n_heads, int head_dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * n_heads * head_dim + seq_idx * n_heads * head_dim + dim_idx;
    int freq_dim_idx = (dim_idx / 2) % (head_dim / 2);
    int freq_idx = seq_idx * head_dim / 2 + freq_dim_idx;

    if (dim_idx % 2 == 0) {
        dst[idx] = __hsub(__hmul(src[idx], cos[freq_idx]), __hmul(src[idx + 1], sin[freq_idx]));
    } else {
        dst[idx] = __hadd(__hmul(src[idx - 1], sin[freq_idx]), __hmul(src[idx], cos[freq_idx]));
    }
}


__global__ void Cat(half* old_cache, half* data, half* new_cache, int seq_len, int dim, int total_len) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * total_len * dim + seq_idx * dim + dim_idx;
    if (seq_idx < total_len - seq_len) {
        new_cache[idx] = old_cache[bs_idx * (total_len - seq_len) * dim + seq_idx * dim + dim_idx];
    } else {
        new_cache[idx] = data[bs_idx * seq_len * dim + (seq_idx - total_len + seq_len) * dim + dim_idx];
    }
}


__global__ void TransposeSeqHeads(half* src, half* dst, int seq_len, int n_heads, int head_dim, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int src_idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;

    int n_heads_idx = dim_idx / head_dim;
    int head_dim_idx = dim_idx % head_dim;
    int dst_idx = bs_idx * seq_len * dim + n_heads_idx * seq_len * head_dim + seq_idx * head_dim + head_dim_idx;

    dst[dst_idx] = src[src_idx];
}


__global__ void TransposeHeadsSeq(half* src, half* dst, int n_heads, int seq_len, int head_dim, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int n_heads_idx = dim_idx / head_dim;
    int head_dim_idx = dim_idx % head_dim;
    int src_idx = bs_idx * seq_len * dim + n_heads_idx * seq_len * head_dim + seq_idx * head_dim + head_dim_idx;
    int dst_idx = bs_idx * seq_len * dim + seq_idx * dim + n_heads_idx * head_dim + head_dim_idx;

    dst[dst_idx] = src[src_idx];
}


__global__ void TransposeSeqHeadDim(half* src, half* dst, int seq_len, int n_heads, int head_dim, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int n_heads_idx = dim_idx / head_dim;
    int head_dim_idx = dim_idx % head_dim;
    int src_idx = bs_idx * seq_len * dim + n_heads_idx * seq_len * head_dim + seq_idx * head_dim + head_dim_idx;
    int dst_idx = bs_idx * seq_len * dim + n_heads_idx * seq_len * head_dim + head_dim_idx * seq_len + seq_idx;

    dst[dst_idx] = src[src_idx];
}


__global__ void MatMul(half* p_src1, half* p_src2, half* p_dst, int seq_len, int total_len, int head_dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * total_len + seq_idx * total_len + dim_idx;
    int src1_idx = bs_idx * seq_len * head_dim + seq_idx * head_dim;
    int src2_idx = bs_idx * total_len * head_dim + dim_idx;

    half sum = __float2half(0.0);
    for (int i = 0; i < head_dim; i++) {
        sum = __hadd(sum, __hmul(p_src1[src1_idx + i], p_src2[src2_idx + i * total_len]));
    }
    p_dst[idx] = sum;
}


__global__ void AddMask(half* p_score, half* p_mask, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int mask_id = 0;
    if (seq_len > 1) {
        mask_id = seq_idx * seq_len + dim_idx;
    }

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    p_score[idx] = __hadd(p_score[idx], p_mask[mask_id]);
}


__global__ void Softmax(half* p_score, int seq_len, int dim) {
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = bs_idx * seq_len * dim + seq_idx * dim;

    half max_val = __float2half(-10000.0);
    for (int i = 0; i < dim; i++) {
        if (p_score[idx + i] > max_val) {
            max_val = p_score[idx + i];
        }
    }
    half sum = __float2half(0.0);
    for (int i = 0; i < dim; i++) {
        sum = __hadd(sum, __float2half(expf(__half2float(__hsub(p_score[idx + i], max_val)))));
    }
    for (int i = 0; i < dim; i++) {
        p_score[idx + i] = __hdiv(__float2half(expf(__half2float(__hsub(p_score[idx + i], max_val)))), sum);
    }
}


__global__ void Add(half* p_data, half* p_shortcut, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    p_data[idx] = __hadd(p_data[idx], p_shortcut[idx]);
}


__global__ void SiLU(half* p_data, int seq_len, int dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = bs_idx * seq_len * dim + seq_idx * dim + dim_idx;
    p_data[idx] = __hmul(p_data[idx], __hdiv(__float2half(1.0), __hadd(__float2half(1.0), __float2half(expf(__half2float(__hneg(p_data[idx])))))));
}


__global__ void Mul(half* p_ff_x1, half* p_ff_x3, int seq_len, int ff_dim) {
    int bs_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;   

    int idx = bs_idx * seq_len * ff_dim + seq_idx * ff_dim + dim_idx;
    p_ff_x1[idx] = __hmul(p_ff_x1[idx], p_ff_x3[idx]);
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
    int max_seq_len = 128;
    int n_heads = 32;
    int head_dim = dim / n_heads;
    int hidden_dim = dim * 4 * 2 / 3;
    int ff_dim = 256 * ((hidden_dim + 256 - 1) / 256);

    int n_elements = bs * seq_len * dim;
    int n_bytes = n_elements * sizeof(half);
    int fp_n_bytes = n_elements * sizeof(float);

    int n_weight_elements = dim * dim;
    int n_weight_bytes = n_weight_elements * sizeof(half);

    int n_output_elements = bs * seq_len * vocab_size;
    int n_output_bytes = n_output_elements * sizeof(half);

    float norm_eps = 1e-6;
    half* h_dim_sqrt_inv = (half*)malloc(sizeof(half));
    h_dim_sqrt_inv[0] = __float2half(1.0 / sqrtf(head_dim));

    int total_len = 0;
    int cache_offset, w_offset, ff_w_offset;

    half* h_input = (half*)malloc(n_bytes);
    half* h_output = (half*)malloc(n_output_bytes);
    half* h_mask = (half*)malloc(seq_len * seq_len * sizeof(half));
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (j > i) {
                h_mask[i * seq_len + j] = __float2half(-10000.0);
            }
            else {
                h_mask[i * seq_len + j] = __float2half(0.0);
            }
        }
    }

    half* h_freq_cos = (half*)malloc(max_seq_len * head_dim * sizeof(half));
    half* h_freq_sin = (half*)malloc(max_seq_len * head_dim * sizeof(half));
    gen_freqs(max_seq_len, head_dim, h_freq_cos, h_freq_sin);

    half* h_embed_w = (half*)malloc(vocab_size * dim * sizeof(half));

    half* h_norm_w = (half*)malloc((n_layers * 2 + 1) * dim * sizeof(half));
    half* h_wq = (half*)malloc(n_layers * n_weight_bytes);
    half* h_wk = (half*)malloc(n_layers * n_weight_bytes);
    half* h_wv = (half*)malloc(n_layers * n_weight_bytes);
    half* h_wo = (half*)malloc(n_layers * n_weight_bytes);

    half* h_ff_w1 = (half*)malloc(n_layers * ff_dim * dim * sizeof(half));
    half* h_ff_w2 = (half*)malloc(n_layers * ff_dim * dim * sizeof(half));
    half* h_ff_w3 = (half*)malloc(n_layers * ff_dim * dim * sizeof(half));

    half* h_w_output = (half*)malloc(vocab_size * dim * sizeof(half));

    std::ifstream fin(weight_path, std::ios::binary);

    fin.read((char*)h_embed_w, vocab_size * dim * sizeof(half));
    fin.read((char*)(h_norm_w + n_layers * 2 * dim), dim * sizeof(half));
    fin.read((char*)h_w_output, vocab_size * dim * sizeof(half));

    #pragma unroll
    for (int i = 0; i < n_layers; i++) {
        fin.read((char*)(h_wq + i * n_weight_elements), n_weight_bytes);
        fin.read((char*)(h_wk + i * n_weight_elements), n_weight_bytes);
        fin.read((char*)(h_wv + i * n_weight_elements), n_weight_bytes);
        fin.read((char*)(h_wo + i * n_weight_elements), n_weight_bytes);

        fin.read((char*)(h_ff_w1 + i * ff_dim * dim), ff_dim * dim * sizeof(half));
        fin.read((char*)(h_ff_w2 + i * ff_dim * dim), ff_dim * dim * sizeof(half));
        fin.read((char*)(h_ff_w3 + i * ff_dim * dim), ff_dim * dim * sizeof(half));

        fin.read((char*)(h_norm_w + i * 2 * dim), 2 * dim * sizeof(half));
    }
    fin.close();

    half* d_data;
    cudaMalloc((void**)&d_data, n_bytes);

    half* d_shortcut;
    cudaMalloc((void**)&d_shortcut, n_bytes);

    float* d_fp_data;
    cudaMalloc((void**)&d_fp_data, fp_n_bytes);

    float* d_fp_power_data;
    cudaMalloc((void**)&d_fp_power_data, fp_n_bytes);

    float* d_fp_mean;
    cudaMalloc((void**)&d_fp_mean, bs * seq_len * sizeof(float));

    half* d_cos;
    cudaMalloc((void**)&d_cos, seq_len * head_dim / 2 * sizeof(half));

    half* d_sin;
    cudaMalloc((void**)&d_sin, seq_len * head_dim / 2 * sizeof(half));

    half* d_dim_sqrt_inv;
    cudaMalloc((void**)&d_dim_sqrt_inv, sizeof(half));
    cudaMemcpy(d_dim_sqrt_inv, h_dim_sqrt_inv, sizeof(half), cudaMemcpyHostToDevice);

    half* d_norm_w;
    cudaMalloc((void**)&d_norm_w, (n_layers * 2 + 1) * dim * sizeof(half));
    cudaMemcpy(d_norm_w, h_norm_w, (n_layers * 2 + 1) *dim * sizeof(half), cudaMemcpyHostToDevice);

    half* d_wq;
    cudaMalloc((void**)&d_wq, n_layers * n_weight_bytes);
    cudaMemcpy(d_wq, h_wq, n_layers * n_weight_bytes, cudaMemcpyHostToDevice);

    half* d_wk;
    cudaMalloc((void**)&d_wk, n_layers * n_weight_bytes);
    cudaMemcpy(d_wk, h_wk, n_layers * n_weight_bytes, cudaMemcpyHostToDevice);

    half* d_wv;
    cudaMalloc((void**)&d_wv, n_layers * n_weight_bytes);
    cudaMemcpy(d_wv, h_wv, n_layers * n_weight_bytes, cudaMemcpyHostToDevice);

    half* d_wo;
    cudaMalloc((void**)&d_wo, n_layers * n_weight_bytes);
    cudaMemcpy(d_wo, h_wo, n_layers * n_weight_bytes, cudaMemcpyHostToDevice);

    half* d_xq;
    cudaMalloc((void**)&d_xq, n_bytes);

    half* d_xq_trans;
    cudaMalloc((void**)&d_xq_trans, n_bytes);

    half* d_xk;
    cudaMalloc((void**)&d_xk, n_bytes);

    half* d_xv;
    cudaMalloc((void**)&d_xv, n_bytes);

    half* d_q_rotary;
    cudaMalloc((void**)&d_q_rotary, n_bytes);

    half* d_k_rotary;
    cudaMalloc((void**)&d_k_rotary, n_bytes);

    half* k_caches;
    cudaMalloc((void**)&k_caches, n_layers * bs * max_seq_len * dim * sizeof(half));

    half* k_caches_trans;
    cudaMalloc((void**)&k_caches_trans, bs * max_seq_len * dim * sizeof(half));

    half* keys;
    cudaMalloc((void**)&keys, bs * max_seq_len * dim * sizeof(half));

    half* v_caches;
    cudaMalloc((void**)&v_caches, n_layers * bs * max_seq_len * dim * sizeof(half));

    half* v_caches_trans;
    cudaMalloc((void**)&v_caches_trans, bs * max_seq_len * dim * sizeof(half));

    half* d_ff_w1;
    cudaMalloc((void**)&d_ff_w1, n_layers * ff_dim * dim * sizeof(half));
    cudaMemcpy(d_ff_w1, h_ff_w1, n_layers * ff_dim * dim * sizeof(half), cudaMemcpyHostToDevice);

    half* d_ff_w2;
    cudaMalloc((void**)&d_ff_w2, n_layers * ff_dim * dim * sizeof(half));
    cudaMemcpy(d_ff_w2, h_ff_w2, n_layers * ff_dim * dim * sizeof(half), cudaMemcpyHostToDevice);

    half* d_ff_w3;
    cudaMalloc((void**)&d_ff_w3, n_layers * ff_dim * dim * sizeof(half));
    cudaMemcpy(d_ff_w3, h_ff_w3, n_layers * ff_dim * dim * sizeof(half), cudaMemcpyHostToDevice);

    half* d_ff_x1;
    cudaMalloc((void**)&d_ff_x1, bs * seq_len * ff_dim * sizeof(half));

    half* d_ff_x3;
    cudaMalloc((void**)&d_ff_x3, bs * seq_len * ff_dim * sizeof(half));

    half* d_output;
    cudaMalloc((void**)&d_output, n_output_bytes);

    half* d_w_output;
    cudaMalloc((void**)&d_w_output, vocab_size * dim * sizeof(half));
    cudaMemcpy(d_w_output, h_w_output, vocab_size * dim * sizeof(half), cudaMemcpyHostToDevice);

    half* d_score;
    cudaMalloc((void**)&d_score, bs * n_heads * seq_len * max_seq_len * sizeof(half));

    half* d_mask;
    cudaMalloc((void**)&d_mask, seq_len * seq_len * sizeof(half));
    cudaMemcpy(d_mask, h_mask, seq_len * seq_len * sizeof(half), cudaMemcpyHostToDevice);

    int next_token = 0;
    half max_prob = 0;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    std::vector<int> results_tokens;
    while (true) {
        #pragma unroll
        for (int i = 0; i < seq_len; i++) {
            #pragma unroll
            for (int j = 0; j < dim; j++) {
                h_input[i * dim + j] = h_embed_w[tokens[i] * dim + j];
            }
        }
        cudaMemcpy(d_data, h_input, bs * seq_len * dim * sizeof(half), cudaMemcpyHostToDevice);

        cudaMemcpy(d_cos, h_freq_cos + total_len * head_dim / 2, seq_len * head_dim / 2 * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin, h_freq_sin + total_len * head_dim / 2, seq_len * head_dim / 2 * sizeof(half), cudaMemcpyHostToDevice);

        total_len += seq_len;

        dim3 dimGrid(dim / 256, seq_len, bs);
        dim3 dimBlock(256, 1, 1);

        dim3 dimCacheGrid(dim / 256, total_len, bs);

        dim3 meanBlock(bs, seq_len);

        dim3 ffGrid(ff_dim / 256, seq_len, bs);
        dim3 ffBlock(256, 1, 1);

        dim3 outGrid(vocab_size / 256, seq_len, bs);
        dim3 outBlock(256, 1, 1);

        dim3 scoreGrid(total_len, seq_len, bs);
        dim3 scoreBlock(1, 1, n_heads);

        dim3 attnGrid(head_dim, seq_len, bs);
        dim3 attnBlock(1, 1, n_heads);

        dim3 softmaxBlock(bs * n_heads, seq_len);

        for (int layer_id = 0; layer_id < n_layers; layer_id++) {
            cache_offset = layer_id * bs * max_seq_len * dim;
            w_offset = layer_id * n_weight_elements;
            ff_w_offset = layer_id * ff_dim * dim;
            // TransformerBlock.shortcut
            Copy<<<dimGrid, dimBlock>>>(d_data, d_shortcut, seq_len, dim);
            // TransformerBlock.attention_norm
            Half2Float<<<dimGrid, dimBlock>>>(d_data, d_fp_data, seq_len, dim);
            Power<<<dimGrid, dimBlock>>>(d_fp_data, d_fp_power_data, seq_len, dim);
            Mean<<<1, meanBlock>>>(d_fp_power_data, d_fp_mean, seq_len, dim);
            AddMean<<<1, meanBlock>>>(d_fp_mean, norm_eps, seq_len);
            Rsqrt<<<1, meanBlock>>>(d_fp_mean, seq_len);
            MulMeanRsqrt<<<dimGrid, dimBlock>>>(d_fp_data, d_fp_mean, seq_len, dim);
            Float2Half<<<dimGrid, dimBlock>>>(d_fp_data, d_data, seq_len, dim);
            MulNormWeight<<<dimGrid, dimBlock>>>(d_data, d_norm_w + 2 * layer_id * dim, seq_len, dim);
            // TransformerBlock.attention
            // qkv
            Linear<<<dimGrid, dimBlock>>>(d_data, d_wq + w_offset, d_xq, seq_len, dim, dim);
            Linear<<<dimGrid, dimBlock>>>(d_data, d_wk + w_offset, d_xk, seq_len, dim, dim);
            Linear<<<dimGrid, dimBlock>>>(d_data, d_wv + w_offset, d_xv, seq_len, dim, dim);
            // rotary_emb
            RotaryEmb<<<dimGrid, dimBlock>>>(d_xq, d_q_rotary, d_cos, d_sin, seq_len, n_heads, head_dim);
            RotaryEmb<<<dimGrid, dimBlock>>>(d_xk, d_k_rotary, d_cos, d_sin, seq_len, n_heads, head_dim);
            // Copy to cache
            Cat<<<dimCacheGrid, dimBlock>>>(k_caches + cache_offset, d_k_rotary, k_caches_trans, seq_len, dim, total_len);
            Cat<<<dimCacheGrid, dimBlock>>>(v_caches + cache_offset, d_xv, v_caches_trans, seq_len, dim, total_len);
            cudaMemcpy(k_caches + cache_offset, k_caches_trans, bs * total_len * dim * sizeof(half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(v_caches + cache_offset, v_caches_trans, bs * total_len * dim * sizeof(half), cudaMemcpyDeviceToDevice);
            // transpose
            TransposeSeqHeads<<<dimGrid, dimBlock>>>(d_q_rotary, d_xq, seq_len, n_heads, head_dim, dim);
            TransposeSeqHeads<<<dimCacheGrid, dimBlock>>>(k_caches + cache_offset, k_caches_trans, total_len, n_heads, head_dim, dim);
            TransposeSeqHeads<<<dimCacheGrid, dimBlock>>>(v_caches + cache_offset, v_caches_trans, total_len, n_heads, head_dim, dim);
            TransposeSeqHeadDim<<<dimCacheGrid, dimBlock>>>(k_caches_trans, keys, total_len, n_heads, head_dim, dim);
            // matmul
            MatMul<<<scoreGrid, scoreBlock>>>(d_xq, keys, d_score, seq_len, total_len, head_dim);
            // score
            MulVal<<<scoreGrid, scoreBlock>>>(d_score, d_dim_sqrt_inv, seq_len, total_len);
            AddMask<<<scoreGrid, scoreBlock>>>(d_score, d_mask, seq_len, total_len);
            // softmax
            Softmax<<<1, softmaxBlock>>>(d_score, seq_len, total_len);
            // matmul
            MatMul<<<attnGrid, attnBlock>>>(d_score, v_caches_trans, d_q_rotary, seq_len, head_dim, total_len);
            // transpose
            TransposeHeadsSeq<<<dimGrid, dimBlock>>>(d_q_rotary, d_k_rotary, n_heads, seq_len, head_dim, dim);
            // linear
            Linear<<<dimGrid, dimBlock>>>(d_k_rotary, d_wo + w_offset, d_data, seq_len, dim, dim);
            // TransformerBlock.add_shortcut
            Add<<<dimGrid, dimBlock>>>(d_data, d_shortcut, seq_len, dim);
            Copy<<<dimGrid, dimBlock>>>(d_data, d_shortcut, seq_len, dim);
            // TransformerBlock.ffn_norm
            Half2Float<<<dimGrid, dimBlock>>>(d_data, d_fp_data, seq_len, dim);
            Power<<<dimGrid, dimBlock>>>(d_fp_data, d_fp_power_data, seq_len, dim);
            Mean<<<1, meanBlock>>>(d_fp_power_data, d_fp_mean, seq_len, dim);
            AddMean<<<1, meanBlock>>>(d_fp_mean, norm_eps, seq_len);
            Rsqrt<<<1, meanBlock>>>(d_fp_mean, seq_len);
            MulMeanRsqrt<<<dimGrid, dimBlock>>>(d_fp_data, d_fp_mean, seq_len, dim);
            Float2Half<<<dimGrid, dimBlock>>>(d_fp_data, d_data, seq_len, dim);
            MulNormWeight<<<dimGrid, dimBlock>>>(d_data, d_norm_w + (2 * layer_id + 1) * dim, seq_len, dim);
            // TransformerBlock.ffn
            Linear<<<ffGrid, ffBlock>>>(d_data, d_ff_w1 + ff_w_offset, d_ff_x1, seq_len, dim, ff_dim);
            Linear<<<ffGrid, ffBlock>>>(d_data, d_ff_w3 + ff_w_offset, d_ff_x3, seq_len, dim, ff_dim);
            SiLU<<<ffGrid, ffBlock>>>(d_ff_x1, seq_len, ff_dim);
            Mul<<<ffGrid, ffBlock>>>(d_ff_x1, d_ff_x3, seq_len, ff_dim);
            Linear<<<dimGrid, dimBlock>>>(d_ff_x1, d_ff_w2 + ff_w_offset, d_data, seq_len, ff_dim, dim);
            // TransformerBlock.add_shortcut
            Add<<<dimGrid, dimBlock>>>(d_data, d_shortcut, seq_len, dim);
        }
        // llama.norm
        Half2Float<<<dimGrid, dimBlock>>>(d_data, d_fp_data, seq_len, dim);
        Power<<<dimGrid, dimBlock>>>(d_fp_data, d_fp_power_data, seq_len, dim);
        Mean<<<1, meanBlock>>>(d_fp_power_data, d_fp_mean, seq_len, dim);
        AddMean<<<1, meanBlock>>>(d_fp_mean, norm_eps, seq_len);
        Rsqrt<<<1, meanBlock>>>(d_fp_mean, seq_len);
        MulMeanRsqrt<<<dimGrid, dimBlock>>>(d_fp_data, d_fp_mean, seq_len, dim);
        Float2Half<<<dimGrid, dimBlock>>>(d_fp_data, d_data, seq_len, dim);
        MulNormWeight<<<dimGrid, dimBlock>>>(d_data, d_norm_w + 2 * n_layers * dim, seq_len, dim);
        // llama.linear
        Linear<<<outGrid, outBlock>>>(d_data, d_w_output, d_output, seq_len, dim, vocab_size);

        cudaMemcpy(h_output, d_output, n_output_bytes, cudaMemcpyDeviceToHost);

        max_prob = __float2half(-10000);
        for (int i = 0; i < vocab_size; i++) {
            if (h_output[(seq_len - 1) * vocab_size + i] > max_prob) {
                max_prob = h_output[bs * (seq_len - 1) * vocab_size + i];
                next_token = i;
            }
        }

        results_tokens.push_back(next_token);

        if (next_token == processor.eos_id()) {
            break;
        }
        else if (total_len >= max_seq_len) {
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
    cudaFree(d_shortcut);
    cudaFree(d_fp_data);
    cudaFree(d_fp_power_data);
    cudaFree(d_fp_mean);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_dim_sqrt_inv);
    cudaFree(d_norm_w);
    cudaFree(d_wq);
    cudaFree(d_wk);
    cudaFree(d_wv);
    cudaFree(d_wo);
    cudaFree(d_xq);
    cudaFree(d_xq_trans);
    cudaFree(d_xk);
    cudaFree(d_xv);
    cudaFree(d_q_rotary);
    cudaFree(d_k_rotary);
    cudaFree(k_caches);
    cudaFree(k_caches_trans);
    cudaFree(keys);
    cudaFree(v_caches);
    cudaFree(v_caches_trans);
    cudaFree(d_ff_w1);
    cudaFree(d_ff_w2);
    cudaFree(d_ff_w3);
    cudaFree(d_ff_x1);
    cudaFree(d_ff_x3);
    cudaFree(d_output);
    cudaFree(d_w_output);
    cudaFree(d_score);
    cudaFree(d_mask);

    free(h_input);
    free(h_output);
    free(h_mask);
    free(h_freq_cos);
    free(h_freq_sin);
    free(h_embed_w);
    free(h_norm_w);
    free(h_wq);
    free(h_wk);
    free(h_wv);
    free(h_wo);
    free(h_ff_w1);
    free(h_ff_w2);
    free(h_ff_w3);
    free(h_w_output);

    return 0;
}