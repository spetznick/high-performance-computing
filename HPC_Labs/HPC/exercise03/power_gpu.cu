// the subroutine for GPU code can be found in several separated text file from
// the Brightspace. You can add these subroutines to this main code.
////////////////////////////////////////////

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda.h"

const int BLOCK_SIZE = 32;  // number of threads per block

// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* h_NormW = NULL;
float* d_NormW = NULL;

// Variables to change
int GlobalSize =
    50;  // this is the dimension of the matrix, GlobalSize*GlobalSize
const float EPS = 0.000005;  // tolerence of the error
int max_iteration = 100;     // the maximum iteration steps
const int n_timers = 2;      //

//
#include <stdio.h>

typedef struct {
    struct timespec start;
    struct timespec end;
    long total_elapsed_ns;  // Accumulated elapsed time in nanoseconds
} CumulativeTimingInfo;

// Function to start the timer
void startTimer(CumulativeTimingInfo* timer) {
    clock_gettime(CLOCK_MONOTONIC, &(timer->start));
}

// Function to stop the timer and accumulate elapsed time
void stopTimer(CumulativeTimingInfo* timer) {
    clock_gettime(CLOCK_MONOTONIC, &(timer->end));
    long elapsed_ns = (timer->end.tv_sec - timer->start.tv_sec) * 1e9 +
                      (timer->end.tv_nsec - timer->start.tv_nsec);
    timer->total_elapsed_ns += elapsed_ns;
}

// Function to print accumulated elapsed time
void printTotalElapsedTime(CumulativeTimingInfo* timer, const char* message) {
    printf("%s Total Elapsed Time: %ld ns\n", message, timer->total_elapsed_ns);
}

// Function to write timing information to a CSV file
void writeTimingInfoToCSV(CumulativeTimingInfo* timeArray,
                          CumulativeTimingInfo* commArray, int n_timers,
                          const char* memory_type) {
    // Write CSV header
    printf("Global Size,Block Size,Time,Comm Time,Memory Type\n");

    // Write timing information to CSV file
    for (int i = 0; i < n_timers; ++i) {
        printf("%d,%d,%ld,%ld,%s\n", GlobalSize, BLOCK_SIZE,
               timeArray[i].total_elapsed_ns, commArray[i].total_elapsed_ns,
               memory_type);
    }
}

// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
float CPUReduce(float*, int);
void Arguments(int, char**);
void checkCardVersion(void);
void print_vec(float* vec, int len);
void print_matrix(float* arr, int len);

// Kernels
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW(float* g_VecW, float* g_NormW, int N);
__global__ void NormalizeW(float* g_VecW, float* g_NormW, float* g_VecV, int N);
__global__ void ComputeLambda(float* g_VecV, float* g_VecW, float* g_Lamda,
                              int N);
// Global memory kernels
__global__ void Global_Av_Product(float* g_MatA, float* g_VecV, float* g_VecW,
                                  int N);
__global__ void Global_FindNormW(float* g_VecW, float* g_NormW, int N);
__global__ void Global_NormalizeW(float* g_VecW, float* g_NormW, float* g_VecV,
                                  int N);
__global__ void Global_ComputeLambda(float* g_VecV, float* g_VecW,
                                     float* g_Lamda, int N);

void CPU_AvProduct() {
    int N = GlobalSize;
    int matIndex = 0;
    for (int i = 0; i < N; i++) {
        h_VecW[i] = 0;
        for (int j = 0; j < N; j++) {
            matIndex = i * N + j;
            h_VecW[i] += h_MatA[matIndex] * h_VecV[j];
        }
    }
}

void CPU_NormalizeW() {
    int N = GlobalSize;
    float normW = 0;
    for (int i = 0; i < N; i++) normW += h_VecW[i] * h_VecW[i];

    normW = sqrt(normW);
    for (int i = 0; i < N; i++) h_VecV[i] = h_VecW[i] / normW;
}

float CPU_ComputeLamda() {
    int N = GlobalSize;
    float lamda = 0;
    for (int i = 0; i < N; i++) lamda += h_VecV[i] * h_VecW[i];

    return lamda;
}

void RunCPUPowerMethod() {
    // printf("*************************************\n");
    float oldLamda = 0;
    float lamda = 0;

    // AvProduct
    CPU_AvProduct();

    // power loop
    for (int i = 0; i < max_iteration; i++) {
        CPU_NormalizeW();
        CPU_AvProduct();
        lamda = CPU_ComputeLamda();
        // printf("CPU lamda at %d: %f \n", i, lamda);
        // If residual is lass than epsilon break
        if (abs(oldLamda - lamda) < EPS) break;
        oldLamda = lamda;
    }
    // printf("*************************************\n");
}

// Host code
int main(int argc, char** argv) {
    // struct timespec t_start, t_end, t_comm_start, t_comm_end;
    // double runtime, comm_time;
    Arguments(argc, argv);

    int N = GlobalSize;
    printf("Matrix size %d X %d \n", N, N);
    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);
    size_t norm_size = sizeof(float);

    CumulativeTimingInfo timeArray[3][n_timers] = {0};
    CumulativeTimingInfo commArray[3][n_timers] = {0};
    const char type1[] = "cpu";
    const char type2[] = "shared";
    const char type3[] = "global";

    // Allocate normalized value in host memory
    h_NormW = (float*)malloc(norm_size);
    // Allocate input matrix in host memory
    h_MatA = (float*)malloc(mat_size);
    // Allocate initial vector V in host memory
    h_VecV = (float*)malloc(vec_size);
    // Allocate W vector for computations
    h_VecW = (float*)malloc(vec_size);

    // Initialize input matrix
    UploadArray(h_MatA, N);
    for (int i = 0; i < n_timers; i++) {
        InitOne(h_VecV, N);
        startTimer(&timeArray[0][i]);
        RunCPUPowerMethod();  // the lamda is already solved here
        stopTimer(&timeArray[0][i]);
    }
    writeTimingInfoToCSV(timeArray[0], commArray[0], n_timers, type1);

    // This is the starting points of GPU
    checkCardVersion();
    // Set the kernel arguments
    int threadsPerBlock = BLOCK_SIZE;
    int sharedMemSize = threadsPerBlock * threadsPerBlock *
                        sizeof(float);  // in per block, the memory is shared
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Allocate matrix and vectors in device memory
    cudaMalloc((void**)&d_MatA, mat_size);
    cudaMalloc((void**)&d_VecV, vec_size);
    cudaMalloc((void**)&d_VecW,
               vec_size);  // This vector is only used by the device
    cudaMalloc((void**)&d_NormW, norm_size);

    for (int j = 0; j < n_timers; j++) {
        startTimer(&timeArray[1][j]);
        // Initialize input matrix
        InitOne(h_VecV, N);

        // Copy from host memory to device memory
        startTimer(&commArray[1][j]);
        cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
        stopTimer(&commArray[1][j]);

        // Power method loops
        float old_lambda = 2 * EPS;
        float lambda = 0;
        *h_NormW = 0.;
        Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_MatA, d_VecV, d_VecW, N);
        cudaDeviceSynchronize();

        for (int i = 0; i < max_iteration; i++) {
            FindNormW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                d_VecW, d_NormW, N);
            cudaDeviceSynchronize();

            // compute sqrt of norm
            startTimer(&commArray[1][j]);
            cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
            *h_NormW = sqrt(*h_NormW);
            cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
            stopTimer(&commArray[1][j]);
            cudaDeviceSynchronize();
            // normalize v
            NormalizeW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                d_VecW, d_NormW, d_VecV, N);
            cudaDeviceSynchronize();

            Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                d_MatA, d_VecV, d_VecW, N);
            cudaDeviceSynchronize();

            cudaMemset(d_NormW, 0, norm_size);  // store result in norm mem
            ComputeLambda<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                d_VecV, d_VecW, d_NormW, N);
            cudaDeviceSynchronize();

            startTimer(&commArray[1][j]);
            cudaMemcpy(&lambda, d_NormW, norm_size, cudaMemcpyDeviceToHost);
            stopTimer(&commArray[1][j]);

            if (abs(old_lambda - lambda) < EPS) break;
            old_lambda = lambda;
        }

        stopTimer(&timeArray[1][j]);
    }
    writeTimingInfoToCSV(timeArray[1], commArray[1], n_timers, type2);

    cudaMemset(d_NormW, 0, norm_size);  // store result in norm mem
    // global memory
    for (int j = 0; j < n_timers; ++j) {
        startTimer(&timeArray[2][j]);
        InitOne(h_VecV, N);

        // Copy from host memory to device memory
        startTimer(&commArray[2][j]);
        cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
        stopTimer(&commArray[2][j]);

        // Power method loops
        float old_lambda = 2 * EPS;
        float lambda = 0;
        *h_NormW = 0.;
        Global_Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_MatA, d_VecV, d_VecW, N);
        cudaDeviceSynchronize();

        for (int i = 0; i < max_iteration; i++) {
            Global_FindNormW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                d_VecW, d_NormW, N);
            cudaDeviceSynchronize();

            // compute sqrt of norm
            startTimer(&commArray[2][j]);
            cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
            *h_NormW = sqrt(*h_NormW);
            cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
            stopTimer(&commArray[2][j]);
            cudaDeviceSynchronize();
            // normalize v
            Global_NormalizeW<<<blocksPerGrid, threadsPerBlock,
                                sharedMemSize>>>(d_VecW, d_NormW, d_VecV, N);
            cudaDeviceSynchronize();

            Global_Av_Product<<<blocksPerGrid, threadsPerBlock,
                                sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
            cudaDeviceSynchronize();

            cudaMemset(d_NormW, 0, norm_size);  // store result in norm mem
            Global_ComputeLambda<<<blocksPerGrid, threadsPerBlock,
                                   sharedMemSize>>>(d_VecV, d_VecW, d_NormW, N);
            cudaDeviceSynchronize();

            startTimer(&commArray[2][j]);
            cudaMemcpy(&lambda, d_NormW, norm_size, cudaMemcpyDeviceToHost);
            stopTimer(&commArray[2][j]);
            // printf("GPU lambda at %d: %5.2f \n", i, lambda);

            if (abs(old_lambda - lambda) < EPS) break;
            old_lambda = lambda;
        }
        stopTimer(&timeArray[2][j]);
    }
    writeTimingInfoToCSV(timeArray[2], commArray[2], n_timers, type3);
    Cleanup();
}

void Cleanup(void) {
    // Free device memory
    if (d_MatA) cudaFree(d_MatA);
    if (d_VecV) cudaFree(d_VecV);
    if (d_VecW) cudaFree(d_VecW);
    if (d_NormW) cudaFree(d_NormW);

    // Free host memory
    if (h_MatA) free(h_MatA);
    if (h_VecV) free(h_VecV);
    if (h_VecW) free(h_VecW);
    if (h_NormW) free(h_NormW);

    exit(0);
}

void print_vec(float* vec, int len) {
    for (size_t i = 0; i < len; i++) {
        printf("%1.2f, ", vec[i]);
    }
    printf("\n");
}

void print_matrix(float* arr, int len) {
    for (size_t i = 0; i < len; i++) {
        for (size_t j = 0; j < len; j++) {
            printf("%1.2f, ", arr[i * len + j]);
        }
        printf("\n");
    }
}

// Allocates an array with zero value.
void InitOne(float* data, int n) {
    for (int i = 0; i < n; i++) data[i] = 0;
    data[0] = 1;
}

void UploadArray(float* data, int n) {
    int total = n * n;
    int value = 1;
    for (int i = 0; i < total; i++) {
        data[i] = (int)(rand() % (int)(101));  // 1;//value;
        value++;
        if (value > n) value = 1;
        // data[i] = 1;
    }
}

// Obtain program arguments
void Arguments(int argc, char** argv) {
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
            GlobalSize = atoi(argv[i + 1]);
            i = i + 1;
        }
        if (strcmp(argv[i], "--max_iteration") == 0 ||
            strcmp(argv[i], "-max_iteration") == 0) {
            max_iteration = atoi(argv[i + 1]);
            i = i + 1;
        }
    }
}

void checkCardVersion() {
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);

    printf("This GPU has major architecture %d, minor %d \n", prop.major,
           prop.minor);
    if (prop.major < 2) {
        fprintf(stderr, "Need compute capability 2 or higher.\n");
        exit(1);
    }
}

//   _    _ _______ _____ _      _____ _________     __     __  __ ______
//   _______ _    _  ____  _____   _____
//  | |  | |__   __|_   _| |    |_   _|__   __\ \   / /    |  \/  |  ____|__ __|
//  |  | |/ __ \|  __ \ / ____| | |  | |  | |    | | | |      | |    | |   \ \_/
//  /     | \  / | |__     | |  | |__| | |  | | |  | | (___
//  | |  | |  | |    | | | |      | |    | |    \   /      | |\/| |  __|    | |
//  |  __  | |  | | |  | |\___ \ | |__| |  | |   _| |_| |____ _| |_   | |     |
//  |       | |  | | |____   | |  | |  | | |__| | |__| |____) |
//   \____/   |_|  |_____|______|_____|  |_|     |_|       |_|  |_|______|  |_|
//   |_|  |_|\____/|_____/|_____/

// This is a list of utility methods that should you use in your code

/*****************************************************************************
This function finds the product of Matrix A and vector V
*****************************************************************************/

// ****************************************************************************************************************************************************/
// parallelization method for the Matrix-vector multiplication as follows:

// each thread handle a multiplication of each row of Matrix A and vector V;

// The share memory is limited for a block, instead of reading an entire row of
// matrix A or vector V from global memory to share memory, a square submatrix
// of A is shared by a block, the size of square submatrix is
// BLOCK_SIZE*BLOCK_SIZE; Thus, a for-loop is used to handle a multiplication of
// each row of Matrix A and vector V step by step. In eacg step, two subvectors
// with size BLOCK_SIZE is multiplied.
//*****************************************************************************************************************************************************/

__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N) {
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    int aBegin = N * BLOCK_SIZE * bx;

    int aEnd = aBegin + N - 1;
    int step = BLOCK_SIZE;

    int bBegin = 0;  // BLOCK_SIZE * bx;
    int bIndex = 0;
    int aIndex = 0;
    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += step, b += step) {
        __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];

        __shared__ float bs[BLOCK_SIZE];

        for (int aa = 0; aa < BLOCK_SIZE; aa += 1) {
            aIndex = a + tx + aa * N;
            if (aIndex < N * N)
                As[tx + aa * BLOCK_SIZE] = g_MatA[aIndex];
            else
                As[tx + aa * BLOCK_SIZE] = 0;
        }

        bIndex = b + tx;
        if (bIndex < N)
            bs[tx] = g_VecV[bIndex];
        else
            bs[tx] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[k + tx * BLOCK_SIZE] * bs[k];
        }  //}
        __syncthreads();
    }

    g_VecW[BLOCK_SIZE * bx + tx] = Csub;
}

/****************************************************
Normalizes vector W : W/norm(W)
****************************************************/
__global__ void FindNormW(float* g_VecW, float* g_NormW, int N) {
    // shared memory size declared at kernel launch
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;

    // For thread ids greater than data space
    if (globalid < N) {
        sdata[tid] = g_VecW[globalid];
    } else {
        sdata[tid] = 0;  // Case of extra threads above N
    }

    // each thread loads one element from global to shared mem
    __syncthreads();

    sdata[tid] = sdata[tid] * sdata[tid];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s = s >> 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }
    // atomic operations:
    if (tid == 0) atomicAdd(g_NormW, sdata[0]);
}

__global__ void NormalizeW(float* g_VecW, float* g_NormW, float* g_VecV,
                           int N) {
    // shared memory size declared at kernel launch
    extern __shared__ float sNormData[];
    unsigned int tid = threadIdx.x;
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) sNormData[0] = g_NormW[0];
    __syncthreads();

    // For thread ids greater than data space
    if (globalid < N) {
        g_VecV[globalid] = g_VecW[globalid] / sNormData[0];
    }
}

__global__ void ComputeLambda(float* g_VecV, float* g_VecW, float* g_Lamda,
                              int N) {
    // shared memory size declared at kernel launch
    extern __shared__ float sdataVW[];
    unsigned int tid = threadIdx.x;
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;

    // For thread ids greater than data space
    if (globalid < N) {
        sdataVW[tid] = g_VecV[globalid] * g_VecW[globalid];
    } else {
        sdataVW[tid] = 0;  // Case of extra threads above N
    }

    // each thread loads one element from global to shared mem
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s = s >> 1) {
        if (tid < s) {
            sdataVW[tid] = sdataVW[tid] + sdataVW[tid + s];
        }
        __syncthreads();
    }
    // atomic operations:
    if (tid == 0) atomicAdd(g_Lamda, sdataVW[0]);
}

// Global memory only

__global__ void Global_Av_Product(float* g_MatA, float* g_VecV, float* g_VecW,
                                  int N) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int aBegin = N * BLOCK_SIZE * bx;
    int aEnd = aBegin + N - 1;
    int step = BLOCK_SIZE;

    int bBegin = 0;
    int bIndex = 0;
    int aIndex = 0;
    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += step, b += step) {
        for (int aa = 0; aa < BLOCK_SIZE; aa += 1) {
            aIndex = a + tx + aa * N;
            if (aIndex < N * N)
                Csub += g_MatA[aIndex] * g_VecV[b + tx + aa * N];
        }
    }

    g_VecW[BLOCK_SIZE * bx + tx] = Csub;
}

__global__ void Global_FindNormW(float* g_VecW, float* g_NormW, int N) {
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalid < N) {
        float val = g_VecW[globalid];
        atomicAdd(g_NormW, val * val);
    }
}

__global__ void Global_NormalizeW(float* g_VecW, float* g_NormW, float* g_VecV,
                                  int N) {
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalid < N) {
        float norm = sqrt(g_NormW[0]);
        g_VecV[globalid] = g_VecW[globalid] / norm;
    }
}

__global__ void Global_ComputeLambda(float* g_VecV, float* g_VecW,
                                     float* g_Lamda, int N) {
    unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalid < N) {
        g_Lamda[0] += g_VecV[globalid] * g_VecW[globalid];
    }
}
