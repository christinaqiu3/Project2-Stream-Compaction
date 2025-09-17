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

### Part 2: Naive GPU Scan Algorithm 
* Implements a naive parallel scan on the GPU using CUDA.
* Iteratively applies scan logic for each depth level (d) in multiple kernel launches.
* Not work-efficient and not in-place.
* Demonstrates basic GPU memory handling and parallel loop structure.

### Part 3: Work-Efficient GPU Scan & Stream Compaction 
* Implements the Blelloch (work-efficient) scan algorithm using the upsweep and downsweep phases.
* Handles non-power-of-two input sizes by padding to the next power of two.
* Adds GPU stream compaction using:
 * Map step (0/1 flags for zero vs. non-zero),
 * Scan on flags,
 * Scatter to final output.
* Much faster and scalable compared to the naive implementation.

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