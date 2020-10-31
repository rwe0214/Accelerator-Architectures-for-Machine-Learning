/* Includes, system */
#include <cstdio>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cstdio>
#include <time.h>
#include <stdio.h>
/* Includes, cuda */
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

using namespace std;

#define get_difftime_ms(t1, t2) ((float) (t2 - t1)) / CLOCKS_PER_SEC * 1000

cudnnHandle_t cudnn_handle;
cublasHandle_t cublas_handle;
const float alpha = 1.0f;
const float beta = 0.0f;

struct Tensor4d
{
    float *data;
    int batch;
    int channel;
    int width;
    int height;
    size_t data_size;
};

struct zeros
{
    float *data;
    size_t data_size;
    zeros(vector<int>dims)
    {
        data_size = accumulate(dims.begin(),
                        dims.end(),
                        1,
                        multiplies<int>());
        vector<float> host_data(data_size);
        for(int i = 0; i < data_size; i++)
            host_data[i] = 0;
        
        cudaMalloc((void**)&data,  data_size * sizeof(float));
        cudaMemcpy(data, host_data.data(), data_size * sizeof(float), 
		    cudaMemcpyHostToDevice);
    }
    ~zeros()
    {
        cudaFree(data);
    }
};

void set_Tensor4d(Tensor4d *ptr, int batch, int channel, int height, int width){
    ptr->batch = batch;
    ptr->channel = channel;
    ptr->height = height;
    ptr->width = width;
    ptr->data_size = batch*channel*height*width;
    ptr->data = (float *)malloc(ptr->data_size * sizeof(float));
    srand(time(NULL));
    for(int i = 0; i < ptr->data_size; i++)
        ptr->data[i] = (float)(rand()%256);
}

Tensor4d Fully_connect(Tensor4d *input, int neuron, string layer_name){
    auto exe_start = chrono::steady_clock::now();
    int k = input->width * input->height * input->channel; 
    float *x_data;
    cudaMalloc((void**)&x_data, input->data_size * sizeof(float));
    cudaMemcpy(x_data, input->data, input->data_size * sizeof(float), 
        cudaMemcpyHostToDevice);

    float *w = (float *)malloc(k*neuron*sizeof(float));
    for (int i = 0; i < neuron; i++) {
        for (int j = 0; j < k; j++)
            w[i * k + j] = (float) (rand() % 20);
    }
    float *w_data;
    cudaMalloc((void**)&w_data, k*neuron*sizeof(float));
    cudaMemcpy(w_data, w, k*neuron*sizeof(float), 
        cudaMemcpyHostToDevice);

    float *y_data;
    cudaMalloc((void**)&y_data, input->batch*k*sizeof(float));

    auto start = chrono::steady_clock::now();
    cublasSgemm(cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            input->batch,
            neuron,
            k,
            &alpha,
            x_data,
            k,
            w_data,
            k,
            &beta,
            y_data,
            1);
    cudaDeviceSynchronize();

    auto end = chrono::steady_clock::now();
    Tensor4d output;
    output.data = (float *)malloc(input->batch*k*sizeof(float));
    cudaMemcpy(output.data, y_data, input->batch*k*sizeof(float), 
        cudaMemcpyDeviceToHost);
    output.batch = 1;
    output.channel = 1;
    output.height = input->batch;
    output.width = neuron;

    auto exe_end = chrono::steady_clock::now();
    int exe_time = static_cast<int>(chrono::duration< double,
                    micro>(exe_end - exe_start).count());
    int fwd_time = static_cast<int>(chrono::duration< double,
        micro>(end - start).count());
    printf("%13s\t\t%8d\t\t%8d\n", layer_name.c_str(), exe_time, fwd_time);
    cudaFree(x_data);
    cudaFree(w_data);
    cudaFree(y_data);
    free(w);
    free(input->data);
    return output;
}

Tensor4d maxpool(Tensor4d *input, int pool_size, int stride, string layer_name)
{
    auto exe_start = chrono::steady_clock::now();
    cudnnTensorDescriptor_t x_desc;
    void *x_data;
    size_t data_size = input->batch * input->channel * input->height * input->width;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        input->batch, input->channel, input->height, input->width);
    cudaMalloc((void**)&x_data, data_size * sizeof(float));
    cudaMemcpy(x_data, input->data, data_size * sizeof(float), 
        cudaMemcpyHostToDevice);

    int out_h, out_w, out_c, out_n;
    vector<int> output_dims_;

    cudnnPoolingDescriptor_t pooling_desc;
    cudnnCreatePoolingDescriptor(&pooling_desc);
    cudnnSetPooling2dDescriptor(pooling_desc, 
                        /*mode=*/CUDNN_POOLING_MAX,
                        /*maxpoolingNanOpt=*/CUDNN_NOT_PROPAGATE_NAN,
                        /*windowHeight=*/pool_size,
                        /*windowWidth=*/pool_size,
                        /*verticalPadding=*/0,
                        /*horizontalPadding=*/0,
                        /*verticalStride=*/stride,
                        /*horizontalStride=*/stride);
    cudnnGetPooling2dForwardOutputDim(pooling_desc,
                        x_desc,
                        &out_n,
                        &out_c,
                        &out_h,
                        &out_w);

    cudnnTensorDescriptor_t y_desc;
    void *y_data;
    data_size = out_n * out_c * out_h * out_w;
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w);
    cudaMalloc((void**)&y_data, data_size * sizeof(float));
    output_dims_ = {out_w, out_h, out_c, out_n};

    auto start = chrono::steady_clock::now();
    cudnnPoolingForward(cudnn_handle,
                pooling_desc,
                &alpha,
                x_desc,
                x_data,
                &beta,
                y_desc,
                y_data);
    cudaDeviceSynchronize();
    auto end = chrono::steady_clock::now();
    
    Tensor4d output;
    output.data = (float *)malloc(data_size * sizeof(float));
    cudaMemcpy(output.data, y_data, data_size * sizeof(float), 
        cudaMemcpyDeviceToHost);
    output.batch = out_n;
    output.channel = out_c;
    output.height = out_h;
    output.width = out_w;
    output.data_size = out_n * out_c * out_h * out_w;

    auto exe_end = chrono::steady_clock::now();
    int exe_time = static_cast<int>(chrono::duration< double,
                    micro>(exe_end - exe_start).count());
    int fwd_time = static_cast<int>(chrono::duration< double,
				    micro>(end - start).count());
    
    printf("%13s\t\t%8d\t\t%8d\n", layer_name.c_str(), exe_time, fwd_time);
    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudaFree(x_data);
    cudaFree(y_data);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    free(input->data);
    return output;
}

Tensor4d convolution(Tensor4d *input, int channels, int num_filters, string layer_name)
{
    auto exe_start = chrono::steady_clock::now();
    cudnnTensorDescriptor_t x_desc;
    void *x_data;
    size_t data_size = input->batch * input->channel * input->height * input->width;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        input->batch, input->channel, input->height, input->width);
    cudaMalloc((void**)&x_data, data_size * sizeof(float));
    cudaMemcpy(x_data, input->data, data_size * sizeof(float), 
        cudaMemcpyHostToDevice);

    cudnnFilterDescriptor_t w_desc;
    void *w_data;
    data_size = num_filters * channels * 3 * 3;
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnSetFilter4dDescriptor(w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        num_filters, channels, 3, 3);
    cudaMalloc((void**)&w_data, data_size * sizeof(float));
    vector<float> random_filter(data_size);
    for(int i = 0; i < data_size; i++)
        random_filter[i] = 0;
    cudaMemcpy(w_data, random_filter.data(), data_size * sizeof(float), 
        cudaMemcpyHostToDevice);
    

    int out_h, out_w, out_c, out_n;
    vector<int> output_dims_;

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor( conv_desc,
                        /*pad_height=*/1,
                        /*pad_width=*/1,
                        /*vertical_stride=*/1,
                        /*horizontal_stride=*/1,
                        /*dilation_height=*/1,
                        /*dilation_width=*/1,
                        /*mode=*/CUDNN_CONVOLUTION,
                        /*computeType=*/CUDNN_DATA_FLOAT);
    cudnnGetConvolution2dForwardOutputDim( conv_desc,
                        x_desc,
                        w_desc,
                        &out_n,
                        &out_c,
                        &out_h,
                        &out_w);

    cudnnTensorDescriptor_t y_desc;
    void *y_data;
    data_size = out_n * out_c * out_h * out_w;
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w);
    cudaMalloc((void**)&y_data, data_size * sizeof(float));

    const int requestAlgoCount = 1;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;

    cudnnFindConvolutionForwardAlgorithm( cudnn_handle,
                        x_desc,
                        w_desc,
                        conv_desc,
                        y_desc,
                        requestAlgoCount,
                        &returnedAlgoCount,
                        &perfResults);
    cudnnConvolutionFwdAlgo_t fwd_algo = perfResults.algo;
    size_t fwd_workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize( cudnn_handle,
                        x_desc,
                        w_desc,
                        conv_desc,
                        y_desc,
                        fwd_algo,
                        &fwd_workspace_size);
                       
    vector<int> u = vector<int>{static_cast<int>
                        (fwd_workspace_size / sizeof(float)), 1};
    zeros fwd_workspace(u);


    auto start = chrono::steady_clock::now();

   // fwd conv
   cudnnStatus_t s = cudnnConvolutionForward(cudnn_handle,
                        &alpha,
                        x_desc,
                        x_data,
                        w_desc,
                        w_data,
                        conv_desc,
                        fwd_algo,
                        fwd_workspace.data,
                        fwd_workspace_size,
                        &beta,
                        y_desc,
                        y_data); 
    cudaDeviceSynchronize();
    
    cudnnActivationDescriptor_t activate_desc;
    cudnnCreateActivationDescriptor(&activate_desc);
    cudnnSetActivationDescriptor(activate_desc,
                        CUDNN_ACTIVATION_RELU,
                        CUDNN_NOT_PROPAGATE_NAN,
                        0.0);
    cudnnActivationForward(cudnn_handle,
                        activate_desc,
                        &alpha,
                        y_desc,
                        y_data,
                        &beta,
                        y_desc,
                        y_data);
    cudaDeviceSynchronize();
    auto end = chrono::steady_clock::now();

    Tensor4d output;
    output.data = (float *)malloc(data_size * sizeof(float));
    cudaMemcpy(output.data, y_data, data_size * sizeof(float), 
        cudaMemcpyDeviceToHost);
    output.batch = out_n;
    output.channel = out_c;
    output.height = out_h;
    output.width = out_w;
    output.data_size = out_n * out_c * out_h * out_w;

    auto exe_end = chrono::steady_clock::now();
    int exe_time = static_cast<int>(chrono::duration< double,
                    micro>(exe_end - exe_start).count());
    int fwd_time = static_cast<int>(chrono::duration< double,
				    micro>(end - start).count());
    printf("%13s\t\t%8d\t\t%8d\n", layer_name.c_str(), exe_time, fwd_time);

    // destroy conv desc
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyActivationDescriptor(activate_desc);
    cudaFree(x_data);
    cudaFree(w_data);
    cudaFree(y_data);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    free(input->data);
    return output;
}


int main()
{
    cudnnCreate(&cudnn_handle);
    cublasCreate(&cublas_handle);
    Tensor4d input, output;
    set_Tensor4d(&input, 1, 3, 224, 224);

    printf("\t\tExecution time(ms)\tConv Forward Algo.\n");

    output = convolution(&input, 3, 64, "conv1_1");
    output = convolution(&output, 64, 64, "conv1_2");
    output = maxpool(&output, 2, 2, "max_pooling_1");

    output = convolution(&output, 64, 128, "conv2_1");
    output = convolution(&output, 128, 128, "conv2_2");
    output = maxpool(&output, 2, 2, "max_pooling_2");

    output = convolution(&output, 128, 256, "conv3_1");
    output = convolution(&output, 256, 256, "conv3_2");
    output = convolution(&output, 256, 256, "conv3_3");
    output = maxpool(&output, 2, 2, "max_pooling_3");

    output = convolution(&output, 256, 512, "conv4_1");
    output = convolution(&output, 512, 512, "conv4_2");
    output = convolution(&output, 512, 512, "conv4_3");
    output = maxpool(&output, 2, 2, "max_pooling_4");

    output = convolution(&output, 512, 512, "conv5_1");
    output = convolution(&output, 512, 512, "conv5_2");
    output = convolution(&output, 512, 512, "conv5_3");
    output = maxpool(&output, 2, 2, "max_pooling_5");
    output = Fully_connect(&output, 4096, "FC_4096");
    output = Fully_connect(&output, 4096, "FC_4096");
    output = Fully_connect(&output, 1000, "FC_1000");

    cublasDestroy(cublas_handle);
    cudnnDestroy(cudnn_handle);
}
