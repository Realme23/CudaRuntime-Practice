#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>

float nextRand() {
    static unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    seed = (seed * 1664525 + 1013904223);
    return ((seed) / 1.0f);
}

int main()
{



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

//Add two 2D matrices
//Parameters: A = Result Matrix, B = Operand 1, C = Operand 2, size = number of elements _per dimension_
__host__ cudaError_t MatrixMultiply(float* A, const float* B, const float* C, int size) {

    size_t bytesSize = size * size * sizeof(float);

    float* deviceA, * deviceB, * deviceC;
    cudaError_t result;
    result = cudaMalloc(&deviceA, bytesSize);
    if (result != cudaSuccess) goto Error;
    result = cudaMalloc(&deviceB, bytesSize);
    if (result != cudaSuccess) goto Error;
    result = cudaMalloc(&deviceC, bytesSize);
    if (result != cudaSuccess) goto Error;

    result = cudaMemcpy(deviceB, B, bytesSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) goto Error;
    result = cudaMemcpy(deviceC, C, bytesSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) goto Error;

    addMatrix << < >> > (deviceA, deviceB, deviceC, size);

    result = cudaMemcpy(A, deviceA, bytesSize, cudaMemcpyDeviceToHost);

Error:
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return result;
}