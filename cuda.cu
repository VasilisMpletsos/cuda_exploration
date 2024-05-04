#include <stdio.h>
#include <time.h>

// Classic CPU function
void AddTwoVectorsCPU(int A[], int B[], int C[], int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// Kernel definition
__global__ void AddTwoVectorsGPU(int A[], int B[], int C[], int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    
    // Get the limit of the current cuda support
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    int threads_per_block = props.maxThreadsPerBlock;
    printf("Maximum threads per block: %d\n", threads_per_block);

    // Size of the vector
    int N = 900000;

    // Create the vectors
    // THIS WAY with pointers we can reach a lot higher numbers without errors with core dumps
    int* A = (int*) malloc(N * sizeof(int));
    int* B = (int*) malloc(N * sizeof(int));
    int* C = (int*) malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        A[i] = 1;
        B[i] = 1;
    }

    // Used to calculate time
    clock_t start, end;
    double cpu_time_used, gpu_time_used, cuda_mem_copy_time_used;

    // Add some values to the vectors
    for (int i = 0; i < N; i++) {
        A[i] = 1;
        B[i] = 1;
    }

    // Initialize the pointers and allocate the memory in the GPU device
    int *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int);
    printf("Going to save %lu MB of vectors\n", size/1000);
    cudaMalloc((void **)&d_A, N * sizeof(int));
    cudaMalloc((void **)&d_B, N * sizeof(int));
    cudaMalloc((void **)&d_C, N * sizeof(int));

    // Copy vectors A and B from host to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    start = clock();
    AddTwoVectorsCPU(A, B, C, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


    int number_of_blocks = (N + (threads_per_block - 1)) / threads_per_block;
    printf("Number of BLOCKS used %d \n", number_of_blocks);
    start = clock();
    // One CUDA core with N threads. THIS WILL THROW AN ERROR if N > 1024 as 3060 has that capacity
    // AddTwoVectorsGPU<<<1, N>>>(d_A, d_B, d_C, N);
    AddTwoVectorsGPU<<<number_of_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    end = clock();
    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // Wait untill all CUDA threads are executed and THEN copy the memory
    cudaDeviceSynchronize();

    start = clock();
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyHostToDevice);
    end = clock();
    cuda_mem_copy_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    float execution_dff = cpu_time_used/gpu_time_used;

    printf("Sum took %f seconds to execute on CPU\n", cpu_time_used);
    printf("Sum took %f seconds to execute on GPU\n", gpu_time_used);
    printf("Cuda memory copy to CPU took %f seconds\n", cuda_mem_copy_time_used);
    printf("Cuda is %f times faster on this execution\n", execution_dff);
    printf("The vector in place 3000 is %d \n", C[3000]);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}