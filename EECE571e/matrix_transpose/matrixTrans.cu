#include <stdio.h> 
#include <stdlib.h>


#define N 1024
/* can't change threads number because
 * this example is a 2-D array, which means
 * each blocks will have 16*16 = 256 threads
 * the max number of threads a block can
 * handle 512 threads, so that's why when
 * THREADS_PER_BLOCK is set to 32, it exceeds
 * the number of threads can support for sm_12
 */
#define THREADS_PER_BLOCK 16

__global__
void transpose(float *in, float *out, int width) {
//void transpose(float* in, float* out, int width) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y; 
    out[tx * width + ty] = in[ty * width + tx];
}

int main(int args, char** vargs) {

    cudaError_t err = cudaSuccess;
    
    const int HEIGHT = N;
    const int WIDTH = N;
    const int SIZE = WIDTH * HEIGHT * sizeof(float); 

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid(N / threadsPerBlock.x, N / threadsPerBlock.y);

    float *M = (float *)malloc(SIZE);

    printf("DEBUG: Size of 'float' type: %lu\n", sizeof(float));
    printf("DEBUG: Footprint total size: %d bytes\n", SIZE);
    

    for (int i = 0; i < HEIGHT * WIDTH; i++) { 
        M[i] = i; 
#if 0
        printf(" %d", i);
        printf(" %f", M[i]);
#endif
    }
#if 0
    printf("DEBUG: \n\t");
    for (int i=0; i<HEIGHTN*WIDTH; i++) {
       printf("%f ", M[i]);
       if ( (i != 0) && (i % N == (N-1)) )
          printf("\n\t");
    } 
    printf("\n");
#else
    printf("DEBUG: \n\t");
    for (int i=0; i<WIDTH; i++) {
       printf("%f ", M[i]);
    } 
    printf("\n");

#endif


    float *Md = NULL;
    err = cudaMalloc((void **)&Md, SIZE);

    float *Bd = NULL;
    err = cudaMalloc((void **)&Bd, SIZE);

    err = cudaMemcpy(Md, M, SIZE, cudaMemcpyHostToDevice);
    
    printf("\nINFO: Launching CUDA kernel: transpose with blocks=%d, threads=%d...", 
              N/threadsPerBlock.x, THREADS_PER_BLOCK);
 
    transpose<<<blocksPerGrid, threadsPerBlock>>>(Md, Bd, N); 
    err = cudaGetLastError();
 
    printf("  Done\n");
   
    err = cudaMemcpy(M, Bd, SIZE, cudaMemcpyDeviceToHost); 
#if 0
    printf("DEBUG: \n\t");
    for (int i=0; i<HEIGHT*WIDTH; i++) {
       printf("%f ", M[i]);
       if ( (i != 0) && (i % N == (N-1)) )
          printf("\n\t");
    } 
    printf("\n");
#else
    printf("DEBUG: \n\t");
    for (int i=0; i<HEIGHT*WIDTH; i++) {
       if ( (i == 0) || (i % N == 0) )
          printf("%f ", M[i]);
    } 
    printf("\n");
#endif
    
    printf("DEBUG: visually checking the results\n");

    free(M);
    err = cudaFree(Md);
    err = cudaFree(Bd);
    err = cudaDeviceReset();
    
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("INFO: Done\n");

    return 0;
}
