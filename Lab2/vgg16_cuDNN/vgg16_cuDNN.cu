#include <cublas_v2.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <time.h>

#include <iostream>

#define get_difftime_ms(t1, t2) ((float) (t2 - t1)) / CLOCKS_PER_SEC * 1000
#define HEIGHT 224
#define WIDTH 224
#define CHANNEL 3

cublasHandle_t cubHandle;
// for cublas dummy constant
const float alpha = 1.0f;
const float beta = 0.0f;

float *image;
float *d_output;
clock_t T;

__global__ void maxpooling(float *output,
                           const float *input,
                           const int width,
                           const int channels)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    // stride = 2, pool size = 2
    int new_width = width / 2;
    int i = thread_id / new_width * 2;
    int j = thread_id % new_width * 2;
    int index = i * width + j;

    for (int c = 0; c < channels; c++) {
        float max = 0;
        if (max < input[index * channels + c])
            max = input[index * channels + c];
        if (max < input[(index + 1) * channels + c])
            max = input[(index + 1) * channels + c];
        if (max < input[(index + width) * channels + c])
            max = input[(index + width) * channels + c];
        if (max < input[(index + width + 1) * channels + c])
            max = input[(index + width + 1) * channels + c];
        output[thread_id * channels + c] = max;
    }
}

__global__ void padding_image(float *input,
                                const float *raw_input,
                                const int width,
                                const int channels)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    // 1 is padding size
    int start_i = thread_id / width - 1;
    int start_j = thread_id % width - 1;
    int per_channel_width = width * width;
    int hidden_width = 3 * 3 * channels + 1;
    int global_offset = thread_id * hidden_width;

    // 3 is filter size
    for (int c = 0; c < channels; c++) {
        int offset = 0;
        for (int i = start_i; i < start_i + 3; i++) {
            if (i < 0 || i == width)
                continue;
            for (int j = start_j; j < start_j + 3; j++) {
                if (j < 0 || j == width)
                    continue;
                input[global_offset + c * 9 + offset] =
                    raw_input[c * per_channel_width + i * width + j];
                offset++;
            }
        }
    }
    input[(thread_id + 1) * hidden_width - 1] = 1;
}

__global__ void padding_fc(float *input,
                             const float *raw_input,
                             const int width,
                             const int channels)
{
    int thread_id = threadIdx.x;
    int size = width * width;

    for (int s = 0; s < size; s++)
        input[thread_id * size + s] = raw_input[s * channels + thread_id];
    if (thread_id == 0)
        input[width * width * channels] = 1;
}

__global__ void padding(float *input,
                          const float *raw_input,
                          const int width,
                          const int channels)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    // 1 is padding size
    int start_i = thread_id / width - 1;
    int start_j = thread_id % width - 1;
    int hidden_width = 3 * 3 * channels + 1;
    int global_offset = thread_id * hidden_width;

    float relu;
    for (int c = 0; c < channels; c++) {
        int offset = 0;
        for (int i = start_i; i < start_i + 3; i++) {
            if (i < 0 || i == width)
                continue;
            for (int j = start_j; j < start_j + 3; j++) {
                offset++;
                if (j < 0 || j == width)
                    continue;
                relu = raw_input[(i * width + j) * channels + c];
                input[global_offset + c * 9 + offset] = relu < 0 ? 0 : relu;
            }
        }
    }
    input[(thread_id + 1) * hidden_width - 1] = 1;
}

void fully_connected(int width, int channels, int num_filters)
{
    int num_weights = (width * width * channels + 1) * num_filters;
    int filter_size = width * width * channels;
    int hidden_width = filter_size + 1;
    float *weights = (float *) malloc(num_weights * sizeof(float));
    for (int i = 0; i < num_filters; i++) {
        for (int j = 0; j < filter_size; j++)
            weights[i * hidden_width + j] = (float) (rand() % 20);
        weights[i * hidden_width + filter_size] = (float) (rand() % 20);
    }

    float *d_input;
    size_t input_size = (width * width * channels + 1) * sizeof(float);
    clock_t t_start = clock();
    cudaMalloc(&d_input, input_size);
    if (width == 1) {
        // previous output vector (channels * 1), expand to ((channels + 1) * 1)
        // with a 1 at last
        float *output = (float *) malloc((channels + 1) * sizeof(float));
        cudaMemcpy(output, d_output, channels * sizeof(float),
                   cudaMemcpyDeviceToHost);
        output[channels] = 1;
        cudaMemcpy(d_input, output, (channels + 1) * sizeof(float),
                   cudaMemcpyHostToDevice);
        free(output);
    } else {
        // only the first fc needs to padding previous output to a vector
        // (width * width * channels)
        // padding, padding size is in padding* function
        padding_fc<<<1, channels>>>(d_input, d_output, width, channels);
        cudaDeviceSynchronize();
    }

    float *d_weights;
    cudaMalloc(&d_weights, num_weights * sizeof(float));
    cudaFree(d_output);
    cudaMalloc(&d_output, num_filters * sizeof(float));
    cublasSetMatrix(hidden_width, num_filters, sizeof(float), weights,
                    hidden_width, d_weights, hidden_width);
    // weights * input = (num_filters * (channels + 1)) * ((channels + 1) * 1),
    // consider vector as matrix
    cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, num_filters,
                hidden_width, &alpha, d_input, 1, d_weights, hidden_width,
                &beta, d_output, 1);

    clock_t t_end = clock();
    printf("FC%d\t\t%8.3f\t\t%8.3f\n", num_filters,
           get_difftime_ms(t_start, t_end), get_difftime_ms(T, t_start));
    free(weights);
    cudaFree(d_input);
    cudaFree(d_weights);
}

void maxpool(int width, int channels)
{
    float *d_temp;
    size_t mem_size = width * width * channels * sizeof(float);
    clock_t t_start = clock();
    cudaMalloc(&d_temp, mem_size);
    cudaMemcpy(d_temp, d_output, mem_size, cudaMemcpyDeviceToDevice);
    cudaFree(d_output);
    cudaMalloc(&d_output, mem_size / 4);
    maxpooling<<<width / 2, width / 2>>>(d_output, d_temp, width, channels);

    cudaDeviceSynchronize();
    clock_t t_end = clock();
    printf("MAXPOOL\t\t%8.3f\t\t%8.3f\n", get_difftime_ms(t_start, t_end),
           get_difftime_ms(T, t_start));
}

/* 
 * input = (width * width)
 * filter = (3 * 3)
 * stride = 1
 * padding = 1
 * output = (width * width)
 */ 
void convolution(int width, int channels, int num_filters)
{
    int num_weights = (3 * 3 * channels + 1) * num_filters;
    int output_size = width * width * num_filters;
    int filter_size = 3 * 3 * channels;
    int hidden_width = 3 * 3 * channels + 1;
    float *weights = (float *) malloc(num_weights * sizeof(float));
    for (int i = 0; i < num_filters; i++) {
        for (int j = 0; j < filter_size; j++)
            weights[j * num_filters + i] = (float) (rand() % 20);
        weights[filter_size * num_filters + i] = (float) (rand() % 20);
    }

    float *d_raw_input;
    float *d_input;
    size_t input_size = width * width * hidden_width * sizeof(float);
    clock_t t_start = clock();
    cudaMalloc(&d_input, input_size);
    // zero padding
    cudaMemset(d_input, 0, input_size);
    // padding, padding size is in padding* function
    if (channels == 3) {
        size_t raw_input_size = width * width * channels * sizeof(float);
        cudaMalloc(&d_raw_input, raw_input_size);
        cudaMemcpy(d_raw_input, image, raw_input_size, cudaMemcpyHostToDevice);
        padding_image<<<width, width>>>(d_input, d_raw_input, width,
                                          channels);
    } else
        padding<<<width, width>>>(d_input, d_output, width, channels);
    cudaDeviceSynchronize();

    float *d_weights;
    cudaMalloc(&d_weights, num_weights * sizeof(float));
    cudaFree(d_output);
    cudaMalloc(&d_output, output_size * sizeof(float));
    cublasSetMatrix(num_filters, hidden_width, sizeof(float), weights,
                    num_filters, d_weights, num_filters);
    // input * weights = ((width * width) * (3 * 3 * channels + 1)) * ((3 * 3 *
    // channels + 1) * num_filters)
    cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, num_filters, width * width,
                hidden_width, &alpha, d_weights, num_filters, d_input,
                hidden_width, &beta, d_output, num_filters);

    clock_t t_end = clock();
    printf("CONV\t\t%8.3f\t\t%8.3f\n", get_difftime_ms(t_start, t_end),
           get_difftime_ms(T, t_start));
    free(weights);
    if (channels == 3)
        cudaFree(d_raw_input);
    cudaFree(d_input);
    cudaFree(d_weights);
}

void create_image(int height, int width, int channel)
{
    int img_size = height * width * channel;
    image = (float *) malloc(img_size * sizeof(float));
    for (int i = 0; i < img_size; i++)
        image[i] = (float) (rand() % 256);
}

int main()
{
    srand(time(NULL));
    create_image(HEIGHT, WIDTH, CHANNEL);

    cublasCreate(&cubHandle);

    // ReLU layers in padding kernel or maxpooling
    printf("\t   Execution time(ms)    Conv Forward Algo.\n");
    T = clock();
    convolution(224, 3, 64);
    convolution(224, 64, 64);
    maxpool(224, 64);

    convolution(112, 64, 128);
    convolution(112, 128, 128);
    maxpool(112, 128);

    convolution(56, 128, 256);
    convolution(56, 256, 256);
    convolution(56, 256, 256);
    maxpool(56, 256);

    convolution(28, 256, 512);
    convolution(28, 512, 512);
    convolution(28, 512, 512);
    maxpool(28, 512);

    convolution(14, 512, 512);
    convolution(14, 512, 512);
    convolution(14, 512, 512);
    maxpool(14, 512);

    fully_connected(7, 512, 4096);
    fully_connected(1, 4096, 4096);
    fully_connected(1, 4096, 1000);

    cublasDestroy(cubHandle);
    free(image);
    return 0;
}