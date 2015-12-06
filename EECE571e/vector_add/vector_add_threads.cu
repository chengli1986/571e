//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>


#define N 512
//
// kernel routine
// 

__global__ void add_threads(int *a, int *b, int *c)
{
   c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

//
// main code
//

//int main(int argc, char **argv)
int main(void)
{
   int *a, *b, *c;
   int *dev_a, *dev_b, *dev_c;
   int size = N * sizeof(int);
   time_t t;
   
   printf("DEBUG: Size of 'int' type: %lu\n", sizeof(int));
   
   srand((unsigned) time(&t));

   // initialise card

   //cutilDeviceInit(argc, argv);
  
   // allocate device copies of a, b, c
   cudaMalloc( (void**)&dev_a, size );
   cudaMalloc( (void**)&dev_b, size );
   cudaMalloc( (void**)&dev_c, size );

   a = (int*)malloc( size ); 
   b = (int*)malloc( size ); 
   c = (int*)malloc( size );
   
   for (int i=0; i < N; ++i)
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
   printf("DEBUG: a[%d]=%d, b[%d]=%d\n",N-1, a[N-1], N-1, b[N-1]);
   
   // copy inputs to device
   cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice ); 
   cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

   printf("INFO: Launching CUDA kernel: add_block with blocks=%d, threads=%d...", 1, N);

   // launch add() kernel with N parallel blocks
   add_threads<<< 1, N >>>( dev_a, dev_b, dev_c );

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
      printf("DEBUG: a[%d]=%d, b[%d]=%d, c[%d]=%d\n", N-1, a[N-1], N-1, b[N-1], N-1, c[N-1]);
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
