//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

//
// kernel routine
// 

__global__ void dot_product(const int *a, const int *b, int *c)
{
   // each thread in a block sharing the memory, temp
   __shared__ int temp[THREADS_PER_BLOCK];
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   temp[threadIdx.x] = a[idx] * b[idx];
   
   __syncthreads();

   if (0 == threadIdx.x) {
      int sum = 0;
      /* iterate over only threads in the block */
      for (int i=0; i<THREADS_PER_BLOCK; ++i)
         sum += temp[i];
      /* Tricks: only works for sm_11... read the simpleAtomicIntrinsics sample */
      atomicAdd( c, sum );
   }
}

//
// main code
//

int main(int argc, char **argv)
{
   int *a, *b, *c;
   int *dev_a, *dev_b, *dev_c;
   int size = N * sizeof(int);
   int result = 0; 
   time_t t;

   // initialise card - legacy code

   //cutilDeviceInit(argc, argv);
  
   srand((unsigned) time(&t));
   printf("DEBUG: Size of 'int' type: %lu\n", sizeof(int));
   printf("DEBUG: Total footprint size: %d bytes\n", size);

   // allocate device copies of a, b, c
   cudaMalloc( (void**)&dev_a, size );
   cudaMalloc( (void**)&dev_b, size );
   cudaMalloc( (void**)&dev_c, sizeof(int) );

   a = (int*)malloc( size ); 
   b = (int*)malloc( size ); 
   c = (int*)malloc( sizeof(int) );
   
   for (int i=0; i<N; i++)
   {
#if 0
      a[i] = rand()%N;
      b[i] = rand()%N;
#else
      a[i] = 5;
      b[i] = 5;
#endif
   }
   printf("DEBUG: a[%d]=%d, b[%d]=%d\n",0, a[0], 0, b[0]);
   printf("DEBUG: a[%d]=%d, b[%d]=%d\n",1, a[1], 1, b[1]);
   
   // copy inputs to device
   cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice ); 
   cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );
   // the bug is lacking of this line... sigh
   cudaMemcpy( dev_c, c, sizeof(int), cudaMemcpyHostToDevice );

   int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
   // launch dot_product() kernel with N parallel blocks
   printf("INFO: Launching CUDA kernel: dot product with blocks=%d, threads=%d...", blocksPerGrid, THREADS_PER_BLOCK);
   
   dot_product<<< blocksPerGrid, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
   //dot_product<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
   
   printf("  Done\n");
   
   printf("DEBUG: c2 is: %d @ %p\n", *c, &c);
   
   // copy device result back to host copy of c
   cudaMemcpy( c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );
   printf("DEBUG: c3 is: %d @ %p\n", *c, &c);
  
#if 1
   //result = 0;
   for (int i=0; i<N; i++)
   {
      result += a[i] * b[i];
   }
   if (fabs(result - *c) < 1e-5)
      printf("INFO: PASS\n");
   else
      printf("ERROR: *** FAILED *** sum=%d\n", result);
#endif
#if 1
      printf("DEBUG: a[0]=%d, b[0]=%d\n", a[0], b[0]);
      printf("DEBUG: a[%d]=%d, b[%d]=%d, c=%d\n", 1, a[1], 1, b[1], *c);
#endif

   cudaFree( dev_a );
   cudaFree( dev_b );
   cudaFree( dev_c );

   free( a ); 
   free( b ); 
   free( c );

   cudaDeviceReset();

   return 0;
}
