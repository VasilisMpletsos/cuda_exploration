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
    if (i < N) // To avoid exceeding array limit
        C[i] = A[i] + B[i];
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
    int N = 600000;

    // Create the vectors
    int A[N], B[N], C[N];

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
    cudaMalloc((void **)&d_A, N * sizeof(int));
    cudaMalloc((void **)&d_B, N * sizeof(int));
    cudaMalloc((void **)&d_C, N * sizeof(int));

    size_t vector_memory = N * sizeof(int);
    printf("Going to save %lu GB of vectors\n", vector_memory / 1000000);

    // Copy vectors A and B from host to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    start = clock();
    AddTwoVectorsCPU(A, B, C, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


    int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
    if (number_of_blocks >= 3500){
        number_of_blocks = 3500;
    }
    printf("Number of BLOCKS used %d \n", number_of_blocks);
    start = clock();
    // One CUDA core with N threads
    // AddTwoVectorsGPU<<<1, N>>>(d_A, d_B, d_C, N);
    AddTwoVectorsGPU<<<number_of_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    end = clock();
    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    start = clock();
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyHostToDevice);
    end = clock();
    cuda_mem_copy_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    float execution_dff = cpu_time_used/gpu_time_used;

    printf("Sum took %f seconds to execute on CPU\n", cpu_time_used);
    printf("Sum took %f seconds to execute on GPU\n", gpu_time_used);
    printf("Cuda memory copy to CPU took %f seconds\n", cuda_mem_copy_time_used);
    printf("Cuda is %f times faster on this execution\n", execution_dff);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}