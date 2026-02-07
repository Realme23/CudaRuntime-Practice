
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>

cudaError_t addWithGPU(float* c, const float* a, const float* b, unsigned int size);
cudaError_t addWithGPUSetup(float *c, const float *a, const float *b, unsigned int size);
cudaError_t addWithCPUScalar(float* c, const float* a, const float* b, unsigned int size);
cudaError_t addWithCPU4Vector(float* c, const float* a, const float* b, unsigned int size);
cudaError_t addWithCPU8Vector(float* c, const float* a, const float* b, unsigned int size);

__global__ void addKernel(float *c, const float *a, const float *b, unsigned int size)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < size)
        c[i] = a[i] + b[i];
}

float nextRand() {
    static unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    seed = (seed * 1664525 + 1013904223);
    return ((seed) / 1.0f);
}

int main()
{
    const int arraySizeRaw = (1 << 28);
    const int arraySize = arraySizeRaw - (arraySizeRaw % 8);
    float* a = new alignas(32) float[arraySize];
    float* b = new alignas(32) float[arraySize];
    float* c = new alignas(32) float[arraySize];

    for (int i = 0; i < arraySize; i++) {
        a[i] = nextRand();
    }

    for (int i = 0; i < arraySize; i++) {
        b[i] = nextRand();
    }

    std::cout << "Begin bench" << '\n';
    std::cout << "Scalars:\n";
    std::chrono::nanoseconds duration;

    auto now = std::chrono::steady_clock().now();
    for (int i = 0; i < 10; i++) {
        addWithCPUScalar(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;

    std::cout << "Duration was: " << duration.count() / 10 << '\n';
    

    std::cout << "Begin bench" << '\n';
    std::cout << "4Vectors:\n";
    now = std::chrono::steady_clock().now();

    for (int i = 0; i < 10; i++) {
        addWithCPU4Vector(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;
    
    std::cout << "Duration was: " << duration.count() / 10 << '\n';

    std::cout << "Begin bench" << '\n';
    std::cout << "8Vectors:\n";
    now = std::chrono::steady_clock().now();

    for (int i = 0; i < 10; i++) {
        addWithCPU8Vector(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;

    std::cout << "Duration was: " << duration.count() / 10 << '\n';


    std::cout << "Begin bench" << '\n';
    std::cout << "CUDA Setup and return:\n";
    now = std::chrono::steady_clock().now();

    for (int i = 0; i < 10; i++) {
        addWithGPUSetup(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;

    std::cout << "Duration was: " << duration.count() / 10 << '\n';
    auto cd_setup = duration;

    std::cout << "Begin bench" << '\n';
    std::cout << "CUDA addition:\n";
    now = std::chrono::steady_clock().now();

    for (int i = 0; i < 10; i++) {
        addWithGPU(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;
    auto cd_add = duration;

    std::cout << "Duration was: " << duration.count() / 10 << '\n';
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete[] a, b, c;

    return 0;
}

cudaError_t addWithCPUScalar(float c[], const float a[], const float b[], unsigned int size) {
    for (volatile unsigned int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
    return cudaSuccess;
}

cudaError_t addWithCPU4Vector(float* c, const float* a, const float* b, unsigned int size) {
    __m128 a_w, b_w, c_w;
    for (unsigned int i = 0; i < size; i += 4) {
        a_w = _mm_load_ps(a + i);
        b_w = _mm_load_ps(b + i);
        c_w = _mm_add_ps(a_w, b_w);
        _mm_stream_ps(c, c_w);
    }
    return cudaSuccess;
}

cudaError_t addWithCPU8Vector(float *c, const float *a, const float *b, unsigned int size) {
    __m256 a_w, b_w, c_w;
    for (unsigned int i = 0; i < size; i+=8) {
        a_w = _mm256_load_ps(a + i);
        b_w = _mm256_load_ps(b + i);
        c_w = _mm256_add_ps(a_w, b_w);
        _mm256_stream_ps(c, c_w);
    }
    return cudaSuccess;
}

cudaError_t addWithGPUSetup(float* c, const float* a, const float* b, unsigned int size) {
    int byteSize = size * sizeof(float);
    float* deviceA = 0, * deviceB = 0, * deviceC = 0;
    cudaError_t result = cudaMalloc(&deviceA, byteSize);
    if (result != cudaSuccess) goto Error;
    result = cudaMalloc(&deviceB, byteSize);
    if (result != cudaSuccess) goto Error;
    result = cudaMalloc(&deviceC, byteSize);
    if (result != cudaSuccess) goto Error;

    result = cudaMemcpy(deviceA, a, byteSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) goto Error;
    result = cudaMemcpy(deviceB, b, byteSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) goto Error;
    


    Error:
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return cudaSuccess;
}

cudaError_t addWithGPU(float* c, const float* a, const float* b, unsigned int size) {
    int byteSize = size * sizeof(float);
    float* deviceA = 0, * deviceB = 0, * deviceC = 0;
    cudaError_t result;
    result = cudaMalloc(&deviceA, byteSize);
    if (result != cudaSuccess) goto Error;
    result = cudaMalloc(&deviceB, byteSize);
    if (result != cudaSuccess) goto Error;
    result = cudaMalloc(&deviceC, byteSize);
    if (result != cudaSuccess) goto Error;

    result = cudaMemcpy(deviceA, a, byteSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) goto Error;
    result = cudaMemcpy(deviceB, b, byteSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) goto Error;

    addKernel<<<(ceil(size / 1024.0)), (1024) >>>(deviceC, deviceA, deviceB, size);

Error:
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return result;
}