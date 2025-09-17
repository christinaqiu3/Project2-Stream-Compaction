#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define BLOCK_SIZE 512

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpsweep(int n, int d, int* data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = 1 << (d + 1);
            int idx = index * stride + (stride - 1);

            if (idx < n) {
                int left = idx - (1 << d);
                data[idx] += data[left];
            }
        }

        __global__ void kernDownsweep(int n, int d, int* data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = 1 << (d + 1);
            int idx = index * stride + (stride - 1);

            if (idx < n) {
                int left = idx - (1 << d);
                int t = data[left];
                data[left] = data[idx];
                data[idx] += t;
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
         // This can be done in place
         // there won't be a case where one thread writes to and another thread reads from the same location in the array.
         // Test non-power-of-two-sized arrays.
         // your intermediate array sizes will need to be rounded to the next power of two.
        void scan(int n, int *odata, const int *idata) {
            int* dev_data;
            int paddedN = 1 << ilog2ceil(n);  // power of two
            size_t byteSize = paddedN * sizeof(int);

            cudaMalloc((void**)&dev_data, byteSize);

            // Copy input to dev_data, pad with 0 if needed for non-power-of-two
            cudaMemset(dev_data, 0, byteSize);
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            timer().startGpuTimer();

            // === UPSWEEP ===
            for (int d = 0; (1 << (d + 1)) <= paddedN; ++d) {
                int numThreads = paddedN / (1 << (d + 1));
                int numBlocks = (numThreads + blockSize - 1) / blockSize;

                if (numThreads > 0) {
                    kernUpsweep << <numBlocks, blockSize >> > (paddedN, d, dev_data);
                    cudaDeviceSynchronize();
                }
            }

            // Set last element to 0 for downsweep
            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));
            cudaDeviceSynchronize();

            // === DOWNSWEEP ===
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; --d) {
                int numThreads = paddedN / (1 << (d + 1));
                int numBlocks = (numThreads + blockSize - 1) / blockSize;

                if (numThreads > 0) {
                    kernDownsweep << <numBlocks, blockSize >> > (paddedN, d, dev_data);
                    cudaDeviceSynchronize();
                }
            }

            timer().endGpuTimer();

            // Copy result back
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
        }

        // copy to remove timer issues
        void scanCopy(int n, int* odata, const int* idata) {
            int* dev_data;
            int paddedN = 1 << ilog2ceil(n);  // power of two
            size_t byteSize = paddedN * sizeof(int);

            cudaMalloc((void**)&dev_data, byteSize);

            // Copy input to dev_data, pad with 0 if needed for non-power-of-two
            cudaMemset(dev_data, 0, byteSize);
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            //timer().startGpuTimer();

            // === UPSWEEP ===
            for (int d = 0; (1 << (d + 1)) <= paddedN; ++d) {
                int numThreads = paddedN / (1 << (d + 1));
                int numBlocks = (numThreads + blockSize - 1) / blockSize;

                if (numThreads > 0)
                    kernUpsweep << <numBlocks, blockSize >> > (paddedN, d, dev_data);
            }

            // Set last element to 0 for downsweep
            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));

            // === DOWNSWEEP ===
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; --d) {
                int numThreads = paddedN / (1 << (d + 1));
                int numBlocks = (numThreads + blockSize - 1) / blockSize;

                if (numThreads > 0)
                    kernDownsweep << <numBlocks, blockSize >> > (paddedN, d, dev_data);
            }

            //timer().endGpuTimer();

            // Copy result back
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */        
        int compact(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_bools;
            int* dev_indices;
            int* dev_odata;

            int paddedN = 1 << ilog2ceil(n);  // power of two

            size_t byteSize = n * sizeof(int);

            cudaMalloc((void**)&dev_idata, byteSize);
            cudaMalloc((void**)&dev_bools, byteSize);
            cudaMalloc((void**)&dev_indices, byteSize);
            cudaMalloc((void**)&dev_odata, byteSize);
            checkCUDAErrorWithLine("cudaMalloc failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            // === Step 1: Map to Boolean ===
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
            checkCUDAErrorWithLine("After kernMapToBoolean");
            cudaDeviceSynchronize();

            // === Step 2: Scan on bools to get indices ===
            scanCopy(n, dev_indices, dev_bools);
            checkCUDAErrorWithLine("After scan");
            cudaDeviceSynchronize();

            // === Step 3: Scatter ===
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAErrorWithLine("After scatter");
            cudaDeviceSynchronize();

            timer().endGpuTimer();


            // Get total number of non-zero elements
            int count = 0;
            cudaMemcpy(&count, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int lastBool = 0;
            cudaMemcpy(&lastBool, dev_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int total = count + lastBool;

            // Copy compacted result back to host
            cudaMemcpy(odata, dev_odata, total * sizeof(int), cudaMemcpyDeviceToHost);

            // Cleanup
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return total;
        }
    }
}
