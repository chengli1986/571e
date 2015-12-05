//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

/* when block=1, threads have to be the
 * the maximum based on current kernel
 * implementations  */
#define N 512
#define THREADS_PER_BLOCK 512

//
// kernel routine
// 

//__global__ void dot_product(const int *a, const int *b, int *c, int numElements)
__global__ void dot_product(const int *a, const int *b, int *c)
{
   // each thread in a block sharing the memory, temp
   __shared__ int temp[N];
   temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
   
   __syncthreads();

   if (threadIdx.x == 0) {
      int sum = 0;
      for (int i=0; i<N; i++)
         sum += temp[i];
      *c = sum;
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
#if 1
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
   
   int threadsPerBlock = THREADS_PER_BLOCK;
   int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

   // launch dot_product() kernel with N parallel threads
   printf("INFO: Launching CUDA kernel: dot product with blocks=%d, threads=%d...", blocksPerGrid, THREADS_PER_BLOCK);
   dot_product<<< blocksPerGrid, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
   
   printf("  Done\n");

   // copy device result back to host copy of c
   cudaMemcpy( c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );
  
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
      //printf("Checking results %d\n", a[0]+b[0]-c[0]);
#endif

 
   free( a ); 
   free( b ); 
   free( c );

   cudaFree( dev_a );
   cudaFree( dev_b );
   cudaFree( dev_c );

   cudaDeviceReset();

   return 0;
}
