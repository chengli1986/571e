#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 512
#define THREADS_PER_BLOCK 16

//
// kernel routine
// 

__global__ void matrix_add(const int *a, const int *b, int *c)
{
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int index = col + row * N;

   if (col < N && row < N)
      c[index] = a[index] + b[index];
}

//
// main code
//

int main(int argc, char **argv)
{

   //int *a, *b, *c;
   int a[N][N], b[N][N], c[N][N];
   int *dev_a, *dev_b, *dev_c;
   int size = N * N * sizeof(int);
   int total;
   
   printf( "DEBUG: Size of 'int' type: %lu\n", sizeof(int) );
   printf( "DEBUG: Total footprint size: %d\n", size );

#if 0   
   // allocate host memory of a, b, c
   a = (int *)malloc( size );
   b = (int *)malloc( size );
   c = (int *)malloc( size );
#endif

   // allocate device copies of a, b, c
   cudaMalloc( (void**)&dev_a, size );
   cudaMalloc( (void**)&dev_b, size );
   cudaMalloc( (void**)&dev_c, size );

   for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
#if 0
      a[i][j] = rand()%N;
      b[i][j] = rand()%N;
      c[i][j] = 0; // init
#else
#if 1
      a[i][j] = 1;
      b[i][j] = 2;
      c[i][j] = 0; // init
#else
      *a = 5; a++;
      *b = 1; b++;
      *c = 0; c++;
#endif
#endif
      }
   }
#if 0
   printf("DEBUG: \n\t");
   for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
         printf("%d ", a[i][j]);
      }
      printf("\n\t");
   } 
   printf("\n\t");

   for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
         printf("%d ", b[i][j]);
      }
      printf("\n\t");
   } 
   printf("\n\t");

   for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
         printf("%d ", c[i][j]);
      }
      printf("\n\t");
   } 
   printf("\n");

#endif
   
   // copy inputs to device
   cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice ); 
   cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );


   dim3 dimBlock ( THREADS_PER_BLOCK, THREADS_PER_BLOCK );
   dim3 dimGrid ( N/THREADS_PER_BLOCK, N/THREADS_PER_BLOCK );
   
   // launch add() kernel with N parallel blocks
   printf("INFO: Launching CUDA kernel: matrix_add with blocks=%d, threads=%d...", 
	N/THREADS_PER_BLOCK, THREADS_PER_BLOCK);
   
   matrix_add<<< dimGrid, dimBlock >>>( dev_a, dev_b, dev_c );
   
   printf("  Done\n");

   // copy device result back to host copy of c
   cudaMemcpy( c, dev_c, size, cudaMemcpyDeviceToHost );
 
#if 0
   printf("\n\t");
   for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
         printf("%d ", c[i][j]);
         total += c[i][j];
      }
      printf("\n\t");
   } 
   printf("\n");
#else
   for (int i=0; i<N; i++) 
      for (int j=0; j<N; j++) 
         total += c[i][j];
#endif

   if ( total == (a[0][0] + b[0][0]) * N * N )
      printf("INFO: PASS: total=%d, c[0][0]=%d\n", total, c[0][0]);
   else
      printf("INFO: FAIL: total=%d, c[0][0]=%d\n", total, c[0][0]);

   cudaFree( dev_a );
   cudaFree( dev_b );
   cudaFree( dev_c );

   cudaDeviceReset();
   
   return 0;
}
