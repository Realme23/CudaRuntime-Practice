
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <iostream>
#include <vector>
#include <chrono>

int main()
{



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::clog << "cudaDeviceReset failed!";
        return -1;
    }
    return 0;
}
