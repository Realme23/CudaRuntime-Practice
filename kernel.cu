
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "BS_thread_pool.hpp"

#include <execution>

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <ranges>

void addWithGPU(float* c, const float* a, const float* b, unsigned int size);
void addWithGPUSetup(float* c, const float* a, const float* b, unsigned int size, float*&, float*&, float*&);
void addWithGPUFinish(float* c, const float* a, const float* b, unsigned int size, float*&, float*&, float*&);
void addWithCPUScalar(float c[], const float a[], const float b[], unsigned int size);
void addWithCPU4Vector(float* c, const float* a, const float* b, unsigned int size);
void addWithCPU8Vector(float* c, const float* a, const float* b, unsigned int size);
void addWithCPUMulti8Vector(float* c, const float* a, const float* b, unsigned int size);

__global__ void addKernel(float* c, const float* a, const float* b, unsigned int size)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[i];
    else
        c[i] = 1;
}

float nextRand() {
    static unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    seed = (seed * 1664525 + 1013904223);
    for (int i = 0; i < (seed % 32); i++)
        seed = (seed * 1664525 + 1013904223);
    return ((seed) / 1.0f);
}

int main()
{
    const int arraySizeRaw = (1 << 28);
    const int arraySize = arraySizeRaw - (arraySizeRaw % (8 * 64));
    float* a = new alignas(32) float[arraySize];
    float* b = new alignas(32) float[arraySize];
    float* c = new alignas(32) float[arraySize];
    float* check = new alignas(32) float[arraySize];

#pragma loop(hint_parallel(0))
#pragma loop(ivdep)
    for (int i = 0; i < arraySize; i++) {
        a[i] = nextRand();
        b[i] = nextRand();
        check[i] = a[i] + b[i];
    }

    std::chrono::steady_clock::time_point now;
    std::chrono::nanoseconds duration;

    std::cout << "Begin bench" << '\n';
    std::cout << "Scalars:\n";


    now = std::chrono::steady_clock().now();
    for (int i = 0; i < 10; i++) {
        addWithCPUScalar(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;

    for (int i = 0; i < arraySize; i++) {
        if (c[i] != check[i])
            std::clog << "Returned values do not match! " << i << " " << c[i] << ", " << check[i];
        c[i] = 0;
    }
    std::cout << "Duration was: " << duration.count() / 10 << '\n';



    std::cout << "Begin bench" << '\n';
    std::cout << "4Vectors:\n";
    now = std::chrono::steady_clock().now();

    for (int i = 0; i < 10; i++) {
        addWithCPU4Vector(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;
    for (int i = 0; i < arraySize; i++) {
        if (c[i] != check[i])
            std::clog << "Returned values do not match! " << i << " " << c[i] << ", " << check[i];
        c[i] = 0;
    }

    std::cout << "Duration was: " << duration.count() / 10 << '\n';

    std::cout << "Begin bench" << '\n';
    std::cout << "8Vectors:\n";
    now = std::chrono::steady_clock().now();

    for (int i = 0; i < 10; i++) {
        addWithCPU8Vector(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;
    for (int i = 0; i < arraySize; i++) {
        if (c[i] != check[i])
            std::clog << "Returned values do not match! " << i << " " << c[i] << ", " << check[i];
        c[i] = 0;
    }
    std::cout << "Duration was: " << duration.count() / 10 << '\n';

    std::cout << "Begin bench" << '\n';
    std::cout << "8Vectors Multithreaded Setup:" << '\n';
    now = std::chrono::steady_clock().now();

    duration = std::chrono::steady_clock().now() - now;
    std::cout << "Duration was: " << duration.count() / 10 << '\n';

    std::cout << "Begin bench" << '\n';
    std::cout << "8Vectors Multithreaded:" << '\n';

    now = std::chrono::steady_clock().now();
    for (unsigned int i = 0; i < 10; i++) {
        addWithCPUMulti8Vector(c, a, b, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;
    for (int i = 0; i < arraySize; i++) {
        if (c[i] != check[i])
            std::clog << "Returned values do not match! " << i << " " << a[i] << " " << b[i] << " " << c[i] << ", " << check[i] << '\n';
        c[i] = 2;
    }
    std::cout << "Duration was: " << duration.count() / 10 << '\n';


    std::cout << "Begin bench" << '\n';
    std::cout << "CUDA Setup and return:\n";
    now = std::chrono::steady_clock().now();

    float* deviceA, * deviceB, * deviceC;
    addWithGPUSetup(c, a, b, arraySize, deviceA, deviceB, deviceC);
    duration = std::chrono::steady_clock().now() - now;

    std::cout << "Duration was: " << duration.count() << '\n';
    auto cd_setup = duration;

    std::cout << "Begin bench" << '\n';
    std::cout << "CUDA addition:\n";
    now = std::chrono::steady_clock().now();
    for (int i = 0; i < 10; i++) {
        addWithGPU(deviceC, deviceA, deviceB, arraySize);
    }
    duration = std::chrono::steady_clock().now() - now;
    addWithGPUFinish(c, a, b, arraySize, deviceA, deviceB, deviceC);

    for (int i = 0; i < arraySize; i++) {
        if (c[i] != check[i])
            std::clog << "Returned values do not match! " << i << " " << std::bit_cast<unsigned int>(c[i]) << ", " << check[i];
        c[i] = 0;
    }

    std::cout << "Duration was: " << duration.count() / 10 << '\n';

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
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

void addWithCPUScalar(float c[], const float a[], const float b[], unsigned int size) {
#pragma loop(no_parallel)
#pragma loop(no_vector)
    for (int i = 0; i < (int)size; i++) {
        c[i] = a[i] + b[i];
    }
    return;
}

void addWithCPU4Vector(float* c, const float* a, const float* b, unsigned const int size) {
    __m128 a_w, b_w, c_w;
#pragma loop(no_parallel)
    for (int i = 0; i < (int)(size / 4); i++) {
        a_w = _mm_load_ps(a + i * 4);
        b_w = _mm_load_ps(b + i * 4);
        c_w = _mm_add_ps(a_w, b_w);
        _mm_stream_ps(c + i * 4, c_w);
    }
    return;
}

void addWithCPU8Vector(float* c, const float* a, const float* b, unsigned const int size) {
    __m256 a_w, b_w, c_w;
#pragma loop(no_parallel)
    for (int i = 0; i < (int)size / 8; i++) {
        a_w = _mm256_load_ps(a + i * 8);
        b_w = _mm256_load_ps(b + i * 8);
        c_w = _mm256_add_ps(a_w, b_w);
        _mm256_stream_ps(c + i * 8, c_w);
    }
    return;
}

void addWithCPUMulti8Vector(float* c, const float* a, const float* b, unsigned int size) {
    __m256 a_w, b_w, c_w;
#pragma loop(hint_parallel(0))
#pragma loop(ivdep)
    for (int i = 0; i < (int)size / 8; i++) {
        a_w = _mm256_load_ps(a + i * 8);
        b_w = _mm256_load_ps(b + i * 8);
        c_w = _mm256_add_ps(a_w, b_w);
        _mm256_stream_ps(c + i * 8, c_w);
    }
    return;
}

void addWithGPUSetup(float* c, const float* a, const float* b, unsigned int size, float*& deviceA, float*& deviceB, float*& deviceC) {
    int byteSize = size * sizeof(float);
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

    return;

Error:
    std::clog << "Cuda Error! Switching off now." << '\n';
    std::terminate();
}

void addWithGPUFinish(float* c, const float* a, const float* b, unsigned int size, float*& deviceA, float*& deviceB, float*& deviceC) {
    int byteSize = size * sizeof(float);
    cudaError_t result = cudaMemcpy(c, deviceC, byteSize, cudaMemcpyDefault);
    if (result != cudaSuccess)
        goto Error;

    return;

Error:
    std::clog << "Cuda Error! Switching off now." << '\n';
    std::terminate();
}

void addWithGPU(float* deviceC, float const* deviceA, float const* deviceB, unsigned int size) {
    addKernel << <ceil(size / 1024), 1024 >> > (deviceC, deviceA, deviceB, size);
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::clog << "Cuda Error! Switching off now." << '\n';
        std::terminate();
    }
    cudaDeviceSynchronize();
    return;
}