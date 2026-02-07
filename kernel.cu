#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>

__host__ cudaError_t MatrixAdd(float* A, const float* B, const float* C, int size);

float nextRand() {
    static unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    seed = (seed * 1664525 + 1013904223);
    return ((seed) / 1.0f);
}

int main()
{
    constexpr int size = 16;
    //Allocate vectors
    float* A = new float[size * size];
    float* B = new float[size * size];
    float* C = new float[size * size];

    //Fill up vectors
    for (int i = 0; i < size * size; i++) {
        A[i] = 0;
        B[i] = 2 * i % 3;
        C[i] = i % 7;
    }

    for (int i = 0; i < size * size; i++) {
        std::cout << C[i] << ' ';
        if (i % size == (size - 1))
            std::cout << '\n';
    }
    std::cout << '\n';

    for (int i = 0; i < size * size; i++) {
        std::cout << B[i] << ' ';
        if (i % size == (size - 1))
            std::cout << '\n';
    }
    std::cout << '\n';

    //Call the function
    cudaError_t cudaStatus = MatrixAdd(A, B, C, size);
    if (cudaStatus != cudaSuccess) {
        std::clog << "Failed to run!\n";
        return -1;
    }

    //Display results
    for (int i = 0; i < size * size; i++) {
        std::cout << A[i] << ' ';
        if (i % size == (size - 1))
            std::cout << '\n';
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

//Device Kernel
__global__ void addMatrix(float* A, const float* B, const float* C, int size) {
    //Which thread is this?
    int threadC = threadIdx.x + (blockIdx.x * blockDim.x);
    //If the thread is > than the matrix, skip
    if (threadC > size * size) {
        return;
    }
    else {
        //Get the i and j for the thread
        int i = threadC % size;
        int j = threadC / size;
        //Add the element for that thread
        A[i + j * size] = B[i + j * size] + C[i + j * size];
    }
}

//Add two 2D matrices
//Parameters: A = Result Matrix, B = Operand 1, C = Operand 2, size = number of elements _per dimension_
__host__ cudaError_t MatrixAdd(float* A, const float* B, const float* C, int size) {

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

    //Add with 256 threads per block and (size * size)/256 blocks for a total of (size * size) threads
    addMatrix<<<ceil(size * size / 256.0),256>>>(deviceA, deviceB, deviceC, size);

    result = cudaMemcpy(A, deviceA, bytesSize, cudaMemcpyDeviceToHost);

Error:
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    return result;
}