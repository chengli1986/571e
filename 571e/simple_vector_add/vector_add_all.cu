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

__global__ void add_block(int *a, int *b, int *c)
{
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   c[idx] = a[idx] + b[idx];
}

//
// main code
//

int main(int argc, char **argv)
{
   int *a, *b, *c;
   int *dev_a, *dev_b, *dev_c;
   int size = N * sizeof(int);
   time_t t;

   // initialise card - legacy code

   //cutilDeviceInit(argc, argv);
  
   srand((unsigned) time(&t));
   printf("DEBUG: Size of 'int' type: %lu\n", sizeof(int));

   // allocate device copies of a, b, c
   cudaMalloc( (void**)&dev_a, size );
   cudaMalloc( (void**)&dev_b, size );
   cudaMalloc( (void**)&dev_c, size );

   a = (int*)malloc( size ); 
   b = (int*)malloc( size ); 
   c = (int*)malloc( size );
   
   for (int i=0; i<N; i++)
   {
      a[i] = rand()%N;
      b[i] = rand()%N;
   }
   printf("DEBUG: a[%d]=%d, b[%d]=%d\n",0, a[0], 0, b[0]);
   printf("DEBUG: a[%d]=%d, b[%d]=%d\n",1, a[1], 1, b[1]);
   
   // copy inputs to device
   cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice ); 
   cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );
   
   // launch add() kernel with N parallel blocks
   printf("INFO: Launching CUDA kernel: add_block with blocks=%d, threads=%d...", N/THREADS_PER_BLOCK, THREADS_PER_BLOCK);
   
   add_block<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
   
   printf("  Done\n");

   // copy device result back to host copy of c
   cudaMemcpy( c, dev_c, size, cudaMemcpyDeviceToHost );
  
#if 1
   for (int i=0; i<N; i++)
   {
      if (fabs(a[i]+b[i]-c[i]) > 1e-5)
      {
         printf("ERROR: *** FAILED ***\n");
         break;
      } else
      {
         if (i == (N -1))
            printf("INFO: PASS\n");
      }
      //printf("Checking results %d\n", a[i]+b[i]-c[i]);
   }
#endif
#if 1
      printf("DEBUG: a[0]=%d, b[0]=%d, c[0]=%d\n", a[0], b[0], c[0]);
      printf("DEBUG: a[%d]=%d, b[%d]=%d, c[%d]=%d\n", 1, a[1], 1, b[1], 1, c[1]);
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
