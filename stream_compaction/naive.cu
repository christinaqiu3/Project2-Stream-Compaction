#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        __global__ void kernNaiveScan(int n, int offset, int* odata, int* idata) {
            if (n <= 0) return;

            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            
            if (index >= (1 << offset)) {
                odata[index] = idata[index - (1 << offset)] + idata[index];
            } else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        // ilog2ceil(n) separate kernel invocations
        // can't generally operate on an array in-place on the GPU
        // create two device arrays. Swap them at each iteration: 
        // read from A and write to B, read from B and write to A, and so on.
        // Be sure to test non - power - of - two - sized arrays.
        void scan(int n, int *odata, const int *idata) {
            int* dev_in;
            int* dev_out;
            int byteSize = n * sizeof(int);

            cudaMalloc((void**)&dev_in, byteSize);
            checkCUDAErrorWithLine("cudaMalloc dev_in failed!");
            cudaMalloc((void**)&dev_out, byteSize);
            checkCUDAErrorWithLine("cudaMalloc dev_out failed!");
            
            cudaMemcpy(dev_in + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice); // shift input right by one
            cudaMemset(dev_in, 0, sizeof(int)); // setting first elem to 0 for exclusive

            int numThreads = 64;
            int numBlocks = (n + numThreads - 1) / numThreads;

            int logn = ilog2ceil(n);

            timer().startGpuTimer();
            for (int i = 0; i < logn; ++i) {
                kernNaiveScan<<<numBlocks, numThreads>>>(n, i, dev_out, dev_in);

                std::swap(dev_in, dev_out);
            }

            timer().endGpuTimer();


            // After logn iterations, the final result is in dev_in (after last swap)
            cudaMemcpy(odata, dev_in, byteSize, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
        }
    }
}
