#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>
#include <tuple>

#define CudaCheck(...) do { PanicOnError(__VA_ARGS__, __FILE__, __LINE__); } while(0)

float nextRand() {
    static unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    //static unsigned int seed = 0;
    seed = (seed * 1664525 + 1013904223);
    return ((seed) / 1.0e+9f);
}

__host__ void PanicOnError(cudaError_t error, const char* File, int Line) {
    if (error != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(error) << "\nOn line: " << Line << ", " << File << std::endl;
        std::terminate();
    }
}

__host__ void Panic(std::string error_message) {
    std::cerr << "Panic called with: " << error_message << std::endl;
    std::terminate();
}

__host__ std::tuple<float*, float*, float*> hostMultiplyMatrixVectorSetup(float *hostInputVector, float* hostInputMatrix, std::size_t size) {
    unsigned long numBytesVector = size * sizeof(float);
    unsigned long numBytesMatrix = size * size * sizeof(float);

    float* deviceInputMatrix, *deviceInputVector, *deviceOutputVector;
    CudaCheck(cudaMalloc(&deviceInputVector, numBytesVector));
    CudaCheck(cudaMalloc(&deviceInputMatrix, numBytesMatrix));
    CudaCheck(cudaMalloc(&deviceOutputVector, numBytesVector));

    CudaCheck(cudaMemcpy(deviceInputVector, hostInputVector, numBytesVector, cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(deviceInputMatrix, hostInputMatrix, numBytesMatrix, cudaMemcpyHostToDevice));
    return std::make_tuple(deviceInputVector, deviceInputMatrix, deviceOutputVector);
}

__host__ void hostMultiplyMatrixVectorClose(float* deviceInputVector, float* deviceInputMatrix, float* deviceOutputVector, float* hostOutputVector, std::size_t size) {
    unsigned long numBytesVector = size * sizeof(float);
    unsigned long numBytesMatrix = size * size * sizeof(float);

    CudaCheck(cudaMemcpy(hostOutputVector, deviceOutputVector, numBytesVector, cudaMemcpyDeviceToHost));

    CudaCheck(cudaFree(deviceInputVector));
    CudaCheck(cudaFree(deviceInputMatrix));
    CudaCheck(cudaFree(deviceOutputVector));
    return;
}

__global__ void MatrixMultiplyKernelDevice(float* deviceInputVector, float* deviceInputMatrix, float* deviceOutputVector, int size) {
    int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    float accumulate = 0;
    if (threadId < size) {
        accumulate = 0;
        for (int j = 0; j < size; j++) {
            int index_ij = j + threadId * size;
            float b_ij = deviceInputMatrix[index_ij];
            int index_j = j;
            float c_j = deviceInputVector[j];
            accumulate += b_ij * c_j;
            //printf("%d %d -> %d: %f * %f = %f, %f\n", threadId, j, index_ij, b_ij, c_j, b_ij * c_j, accumulate);
        }
        deviceOutputVector[threadId] = accumulate;
    }
}

template<unsigned int sizeBlocks>
__host__ void hostMultiplyMatrixVectorCall(float* deviceInputVector, float* deviceInputMatrix, float* deviceOutputVector, std::size_t size) {
    MatrixMultiplyKernelDevice<<<ceil(size*1.0/sizeBlocks), sizeBlocks>>>(deviceInputVector, deviceInputMatrix, deviceOutputVector, size);
    CudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();
    CudaCheck(cudaGetLastError());
}

__host__ void fillFloats(float* input, std::size_t size) {
    for (int i = 0; i < size; i++) {
        input[i] = nextRand();
    }
}

__host__ void showVector(float* input, std::size_t size) {
    std::cout << "Vector:\n";
    for (int i = 0; i < size; i++) {
        std::cout << input[i] << ' ';
    }
    std::cout << "\n";
}

__host__ void showMatrix(float* input, std::size_t size) {
    std::cout << "Matrix:\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << input[i*size + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << "\n";
}

int main()
{
    constexpr int size = 3;
    float* hostInputVector = new alignas(32) float[size];
    float* hostInputMatrix = new alignas(32) float[size * size];
    float* hostOutputVector = new alignas(32) float[size];

    fillFloats(hostInputMatrix, size * size);
    fillFloats(hostInputVector, size);
    showMatrix(hostInputMatrix, std::min(size, 10));
    showVector(hostInputVector, std::min(size, 10));

    auto deviceResults = hostMultiplyMatrixVectorSetup(hostInputVector, hostInputMatrix, size);
    float* deviceInputVector = std::get<0>(deviceResults);
    float* deviceInputMatrix = std::get<1>(deviceResults);
    float* deviceOutputVector = std::get<2>(deviceResults);

    hostMultiplyMatrixVectorCall<128>(deviceInputVector, deviceInputMatrix, deviceOutputVector, size);
    hostMultiplyMatrixVectorClose(deviceInputVector, deviceInputMatrix, deviceOutputVector, hostOutputVector, size);
    showVector(hostOutputVector, std::min(size, 10));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CudaCheck(cudaDeviceReset());

    return 0;
}
