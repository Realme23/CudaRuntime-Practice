
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void matrixVectorMultiply(float* VectorResult, const float* Matrix, const float* Vector, unsigned size) {
    const int idx = (threadIdx.x) + (blockIdx.x * blockDim.x);
    //Only if within limits
    if(idx < size)
        //Dot product the row with the vector
        for (int i = 0; i < size; i++) {
            VectorResult[idx] += Matrix[i * size + idx] * Vector[i];
        }
}


__host__ cudaError_t matrixVectorMultiplyHost(float* VectorResult, const float* Matrix, const float* Vector, unsigned size) {
    float* VectorDevice = 0, * MatrixDevice = 0, * ResultDevice = 0;
    const int byteSizeVector = size * sizeof(float);
    const int byteSizeMatrix = size * size * sizeof(float);

    cudaError_t cudaresult = cudaMalloc(&VectorDevice, byteSizeVector);
    if (cudaresult != cudaSuccess) { goto Error; }
    cudaresult = cudaMalloc(&MatrixDevice, byteSizeMatrix);
    if (cudaresult != cudaSuccess) { goto Error; }
    cudaresult = cudaMalloc(&ResultDevice, byteSizeVector);
    if (cudaresult != cudaSuccess) { goto Error; }

    cudaresult = cudaMemcpy(VectorDevice, Vector, byteSizeVector, cudaMemcpyHostToDevice);
    if (cudaresult != cudaSuccess) { goto Error; }
    cudaresult = cudaMemcpy(MatrixDevice, Matrix, byteSizeMatrix, cudaMemcpyHostToDevice);
    if (cudaresult != cudaSuccess) { goto Error; }

    matrixVectorMultiplyT << <ceil(size / 256.0), 256 >> > (ResultDevice, MatrixDevice, VectorDevice, size);

    cudaresult = cudaMemcpy(VectorResult, ResultDevice, byteSizeVector, cudaMemcpyDeviceToHost);
    if (cudaresult != cudaSuccess) { goto Error; }

Error:
    cudaFree(VectorDevice);
    cudaFree(MatrixDevice);
    cudaFree(ResultDevice);
    return cudaresult;
}

int main()
{
    const int size = 10;

    float* Vector, * Matrix, * Result;
    Vector = new float[size];
    Matrix = new float[size * size];
    Result = new float[size];

    for (int i = 0; i < size; i++) {
        Vector[i] = (1234 * i + i * i + 3) % 10;
        for(int j = i * size; j < (i + 1) * size; j++)
            Matrix[j] = (i *j + 3) % 5;
        Result[i] = 0;
    }

    for (int i = 0; i < size && i < 10; i++) {
        std::cout << Vector[i] << '\n';
    }
    std::cout << "\n\n";
    for (int i = 0; i < size && i < 10; i++) {
        for (int j = 0; j < size && j < 10; j++) {
            std::cout << Matrix[i*size + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    cudaError_t cudaStatus = matrixVectorMultiplyHost(Result, Matrix, Vector, size);
    if (cudaStatus != cudaSuccess) {
        std::clog << "MatrixVectorMultiply failed!\n";
        return -1;
    }

    for (int i = 0; i < size && i < 10; i++) {
        std::cout << Result[i] << ' ';
    }
    std::cout << '\n';

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::clog << "cudaDeviceReset failed!";
        return -1;
    }
    return 0;
}
