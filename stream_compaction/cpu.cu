#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = idata[i - 1] + odata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            
            int* map = new int[n];
            int* scanArr = new int[n];
            timer().startCpuTimer();

            // MAP
            for (int i = 0; i < n; i++) {
                map[i] = (idata[i] != 0) ? 1 : 0;
            }

            // SCAN
            scanArr[0] = 0;
            for (int i = 1; i < n; i++) {
                scanArr[i] = scanArr[i - 1] + map[i - 1];
            }

            // SCATTER
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (map[i] != 0 && scanArr[i] < n) {
                    odata[scanArr[i]] = idata[i];
                    count++;
                }
            }
            delete[] map;
            delete[] scanArr;

            timer().endCpuTimer();
            return count;
        }
    }
}
