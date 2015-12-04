#include <stdio.h> 
#include <stdlib.h>

#define N 1024
#define THREADS_PER_BLOCK 16

__global__
void transpose(float* in, float* out, int width) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y; 
    out[tx * width + ty] = in[ty * width + tx];
}

int main(int args, char** vargs) {
    const int HEIGHT = N;
    const int WIDTH = N;
    const int SIZE = WIDTH * HEIGHT * sizeof(float); 
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid(N / threadsPerBlock.x, N / threadsPerBlock.y);

    float* M = (float*)malloc(SIZE);

    printf("DEBUG: Size of 'float' type: %lu\n", sizeof(float));
    

    for (int i = 0; i < HEIGHT * WIDTH; i++) { 
        M[i] = i; 
    }

    printf("DEBUG: ");
    for (int i=0; i<10; i++) {
       printf("%f ", M[i]);
    } 
    printf("\n");


    float* Md = NULL;
    cudaMalloc((void**)&Md, SIZE);

    cudaMemcpy(Md,M, SIZE, cudaMemcpyHostToDevice);
    
    float* Bd = NULL;
    cudaMalloc((void**)&Bd, SIZE);
   
    printf("INFO: Launching CUDA kernel: transpose with blocks=%d, threads=%d...", 
              16, 1024/16);
 
    /* 64 blocks, 16 threads */ 
    transpose<<<blocksPerGrid, threadsPerBlock>>>(Md, Bd, WIDTH); 
 
    printf("  Done\n");
   
    cudaMemcpy(M, Bd, SIZE, cudaMemcpyDeviceToHost); 

    printf("DEBUG: ");

    for (int i=0; i<10; i++) {
       printf("%f ", M[i]);
    } 
    printf("\n");

    free(M);
    cudaFree(Md);
    cudaFree(Bd);
    cudaDeviceReset();

    return 0;
}
