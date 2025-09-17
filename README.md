CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Christina Qiu
  * [LinkedIn](https://www.linkedin.com/in/christina-qiu-6094301b6/), [personal website](https://christinaqiu3.github.io/), [twitter](), etc.
* Tested on: Windows 11, Intel Core i7-13700H @ 2.40GHz, 16GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (Personal laptop)

## Overview

This is an implementation of scan (prefix sum) and stream compaction algorithms on both the CPU and GPU using CUDA. 

This project includes four parts:

### Part 1: CPU Scan & Stream Compaction 
* Implements a basic exclusive prefix sum (scan) on the CPU.
* Two stream compaction implementations:
 * Without scan: simple loop that filters non-zero values.
 * With scan and scatter: mimics the parallel approach by mapping, scanning, and scattering.
* Used for correctness testing and performance comparison against GPU implementations.
* Runtime O(n)

### Part 2: Naive GPU Scan Algorithm 
* Implements a naive parallel scan on the GPU using CUDA.
* Iteratively applies scan logic for each depth level (d) in multiple kernel launches.
* Not work-efficient and not in-place.
* Demonstrates basic GPU memory handling and parallel loop structure.
* Runtime O(log n) kernel launches, total work O(n log n)
 * At each iteration, a full kernel with n threads is launched

### Part 3: Work-Efficient GPU Scan & Stream Compaction 
* Implements the Blelloch (work-efficient) scan algorithm using the upsweep and downsweep phases.
* Handles non-power-of-two input sizes by padding to the next power of two.
* Adds GPU stream compaction using:
 * Map step (0/1 flags for zero vs. non-zero),
 * Scan on flags,
 * Scatter to final output.
* Much faster and scalable compared to the naive implementation.
* Runtime O(log n) kernel launches, total work O(n)
 * This is because each kernel does fewer threads of work as d increases.
 * At step (d = 0), launched (threads = n/2), per thread (work = constant)
 * At step (d = 1), launched (threads = n/4), per thread (work = constant)
 * At step (d = logn-1), launched (threads = 1), per thread (work = constant)
 * Thus the total work across all kernels: (n/2) + (n/4) + (n/8) + ... + 1 = O(n)

### Part 4: Using Thrust's Implementation
* Leverages Thrust, a high-performance parallel algorithms library built on CUDA.
* Implements scan using thrust::exclusive_scan.
* Simplifies GPU programming and enables performance benchmarking against custom implementations.

### Output of Scan & Stream Compaction Tests
(Array Size = 2^24)
```
****************
** SCAN TESTS **
****************
    [   7  32  41  34  34  45  20  39  29  27   7  38  38 ...  10   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 28.6697ms    (std::chrono Measured)
    [   0   7  39  80 114 148 193 213 252 281 308 315 353 ... 410882306 410882316 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 30.8242ms    (std::chrono Measured)
    [   0   7  39  80 114 148 193 213 252 281 308 315 353 ... 410882253 410882290 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 14.7507ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 14.6412ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 6.67194ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 6.49674ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.34758ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.26874ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   2   2   1   2   2   3   0   0   3   3   3   3 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 38.4214ms    (std::chrono Measured)
    [   2   2   1   2   2   3   3   3   3   3   2   2   2 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 38.1998ms    (std::chrono Measured)
    [   2   2   1   2   2   3   3   3   3   3   2   2   2 ...   3   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 88.2723ms    (std::chrono Measured)
    [   2   2   1   2   2   3   3   3   3   3   2   2   2 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 11.8516ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 11.6095ms    (CUDA Measured)
    passed
```

## Runtime and Performance Analysis

(note: I tested Array Sizes up to 2^24 because 2^25 is where computation breaks down on my laptop)

1. Hypothesis: The GPU implementations of scan (Naive, Work-Efficient, and Thrust) will outperform the serial CPU scan as the array size increases. The Thrust library implementation will be the fastest on large data sizes due to its highly optimized CUDA backend. The naive GPU scan will be slower than the work-efficient implementation due to redundant work and synchronization overhead.

### Size of Array v. Runtime of Scan (Power of 2) Graph

blockSize = 128

![Data](images/graph1_t.png)
![Graph](images/graph1_v.png)

### Size of Array v. Runtime of Scan (Non Power of 2) Graph

blockSize = 128

![Data](images/graph2_t.png)
![Graph](images/graph2_v.png)

Conclusion: 

CPU scan runtime increases exponentially as the array size increases, due to larger data sizes needing to access memory beyond the L1/L2 caches. 

For smaller inputs, Naive GPU scan can actually outperform the more complex approaches because it uses fewer threads and has less overhead. However, as the input size grows, the naive approach becomes inefficient due to poor memory access patterns. Specifically, threads must access data locations increasingly farther apart, resulting in uncoalesced memory transactions that hurt bandwidth and performance. Additionally, many threads become idle in later stages, reducing hardware utilization.

Work Efficient GPU scan performs two kernels per iteration (upsweep and downsweep). Despite more frequent kernel invocations, this method excels for larger inputs because it minimizes idle threads and accesses memory in a way that favors coalescing. The upsweep and downsweep phases ensure that each element is processed only a logarithmic number of times with well-structured memory access, improving throughput and scaling much better than the naive approach. As a result, this approach ultimately surpasses both the naive GPU scan and the serial CPU scan in speed for large array sizes.

##

2. Hypothesis: The CPU implementation of stream compaction is expected to become significantly slower as the array size increases due to its sequential nature and growing cache/memory pressure. On the other hand, the work-efficient GPU implementation should demonstrate much better scalability, especially on larger arrays, due to parallel processing and improved memory access patterns.

### Size of Array v. Runtime of Stream Compaction (Power of 2) Graph

blockSize = 128

![Data](images/graph3_t.png)
![Graph](images/graph3_v.png)

### Size of Array v. Runtime of Stream Compaction (Non Power of 2) Graph

blockSize = 128

![Data](images/graph4_t.png)
![Graph](images/graph4_v.png)

Conclusion: As expected, the CPU implementation’s runtime increases rapidly with array size (particularly beyond 2^21). In contrast, the work-efficient GPU implementation shows much better scaling, with runtimes increasing much more slowly as array size grows. For smaller sizes (e.g. 2^18 - 2^20), the CPU is actually slightly faster, likely due to lower kernel launch overhead. But this quickly flips as data size increases. This shows why GPU acceleration is critical for real-time or high-volume applications involving stream compaction.

## 

2. Hypothesis:

### Blocksize v. Runtime of Scan Method (Power of 2) Graph

Array Size = 2^20

![Data](images/graph1_t.png)
![Graph](images/graph1_v.png)

### Blocksize v. Runtime of Scan (Non Power of 2) Graph

Array Size = 2^20

![Data](images/graph2_t.png)
![Graph](images/graph2_v.png)

Conclusion: 


