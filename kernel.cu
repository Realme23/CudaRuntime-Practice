#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>
#include <tuple>

float nextRand() {
    static unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    seed = (seed * 1664525 + 1013904223);
    return ((seed) / 1.0f);
}

__host__ void Panic(std::string error_message) {
    std::cerr << "Panic called with: " << error_message << std::endl;
    std::terminate();
}

__host__ void CudaMallocOrPanic(float*& target, std::size_t size) {
    cudaError_t result = cudaMalloc(&target, (unsigned long)size);
    if (result != cudaSuccess) {
        Panic("CudaMalloc failed!");
    }
    return;
}

__host__ void CudaMemcpyOrPanic(float*& source, float*& target, std::size_t size, cudaMemcpyKind kind) {
    cudaError_t result = cudaMemcpy(&target, &source, (unsigned long)size, kind);
    if (result != cudaSuccess) {
        Panic("CudaMemcpy failed!");
    }
    return;
}

__host__ std::tuple<float*, float*, float*> hostMultiplyMatrixVectorSetup(float *hostInputVector, float* hostInputMatrix, std::size_t size) {
    unsigned long numBytesVector = size * sizeof(float);
    unsigned long numBytesMatrix = size * size * sizeof(float);

    float* deviceInputMatrix, *deviceInputVector, *deviceOutputVector;
    CudaMallocOrPanic(deviceInputVector, numBytesVector);
    CudaMallocOrPanic(deviceInputMatrix, numBytesMatrix);
    CudaMallocOrPanic(deviceOutputVector, numBytesVector);

    CudaMemcpyOrPanic(hostInputVector, deviceInputVector, numBytesVector, cudaMemcpyHostToDevice);
    CudaMemcpyOrPanic(hostInputMatrix, deviceInputMatrix, numBytesMatrix, cudaMemcpyHostToDevice);
    return std::make_tuple(deviceOutputVector, deviceInputMatrix, deviceOutputVector);
}

__host__ void CudaFreeOrPanic(float* devicePointer) {
    cudaError_t cudaResult = cudaFree(devicePointer);
    if (cudaResult != cudaSuccess) {
        Panic("CudaFree failed!");
    }
}

__host__ void hostMultiplyMatrixVectorClose(float* deviceInputVector, float* deviceInputMatrix, float* deviceOutputVector, float* hostOutputVector, std::size_t size) {
    unsigned long numBytesVector = size * sizeof(float);
    unsigned long numBytesMatrix = size * size * sizeof(float);

    CudaMemcpyOrPanic(deviceOutputVector, hostOutputVector, numBytesVector, cudaMemcpyDeviceToHost);

    CudaFreeOrPanic(deviceInputVector);
    CudaFreeOrPanic(deviceInputMatrix);
    CudaFreeOrPanic(deviceOutputVector);
    return;
}

__global__ void MatrixMultiplyKernelDevice(float* deviceInputVector, float* deviceInputMatrix, float* deviceOutputVector, int size) {
    int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    float accumulate = 0;
    if (threadId < size) {
        for (int j = 0; j < size; j++) {
            accumulate += deviceInputMatrix[threadId * size + j];
            accumulate += deviceInputVector[j];
            deviceOutputVector[threadId] = accumulate;
        }
    }
}

template<unsigned int sizeBlocks>
__host__ void hostMultiplyMatrixVectorCall(float* deviceInputVector, float* deviceInputMatrix, float* deviceOutputVector, std::size_t size) {
    MatrixMultiplyKernelDevice<<<ceil(size*1.0/sizeBlocks), sizeBlocks>>>(deviceInputVector, deviceInputMatrix, deviceOutputVector, size);
}

__host__ void fillFloats(float* input, std::size_t size) {
    for (int i = 0; i < size; i++) {
        input[i] = nextRand();
    }
}

__host__ void showFloats(float* input, std::size_t size) {
    std::cout << "Vector:\n";
    for (int i = 0; i < size; i++) {
        std::cout << input[i] << ' ';
    }
    std::cout << "\n";
}

int main()
{
    constexpr int size = 32;
    float* hostInputVector = new alignas(32) float[size];
    float* hostInputMatrix = new alignas(32) float[size * size];
    float* hostOutputVector = new alignas(32) float[size];

    fillFloats(hostInputMatrix, size * size);
    fillFloats(hostInputVector, size);
    
    auto deviceResults = hostMultiplyMatrixVectorSetup(hostInputVector, hostInputMatrix, size);
    float* deviceInputVector = std::get<0>(deviceResults);
    float* deviceInputMatrix = std::get<1>(deviceResults);
    float* deviceOutputVector = std::get<2>(deviceResults);

    hostMultiplyMatrixVectorCall<128>(hostInputVector, hostInputMatrix, hostOutputVector, size);
    hostMultiplyMatrixVectorClose(deviceOutputVector, deviceInputMatrix, deviceOutputVector, hostOutputVector, size);
    showFloats(hostOutputVector, std::min(size, 10));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
