#include <stdio.h> 
#include <stdlib.h>

__global__
void transpose(float* in, float* out, int width) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y; 
    out[tx * width + ty] = in[ty * width + tx];
}

int main(int args, char** vargs) {
    const int HEIGHT = 1024;
    const int WIDTH = 1024;
    const int SIZE = WIDTH * HEIGHT * sizeof(float); 
    dim3 bDim(16, 16);
    dim3 gDim(WIDTH / bDim.x, HEIGHT / bDim.y);
    float* M = (float*)malloc(SIZE);
    for (int i = 0; i < HEIGHT * WIDTH; i++) { 
        M[i] = i; 
    }
    float* Md = NULL;
    cudaMalloc((void**)&Md, SIZE);
    cudaMemcpy(Md,M, SIZE, cudaMemcpyHostToDevice);
    
    float* Bd = NULL;
    cudaMalloc((void**)&Bd, SIZE);
    
    transpose<<<gDim, bDim>>>(Md, Bd, WIDTH); 

    cudaMemcpy(M,Bd, SIZE, cudaMemcpyDeviceToHost); 

    return 0;
}
