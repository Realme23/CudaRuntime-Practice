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
