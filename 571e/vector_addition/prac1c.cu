/*
* Two vector addition using CUDA
* based on prac1b.cu
* Modified by: Aryya Dwisatya W - 13512043
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "cutil_inline.h"


//
// kernel routine
// 

/* Initiatin first vector with value of threads id */
__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x;
}

/* Initiatin second vector with value of threads id */
__global__ void my_second_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x; /* udah dengan konstanta jika ingin */
}

/* Adding the value of second vector to the first vector */
__global__ void add_vector(float *x,float *y)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] += y[tid];
}
//
// main code
//

int main(int argc, char **argv)
{
  float *h_x, *d_x,*d_x2;
  int   nblocks, nthreads, nsize, n; 

  // initialise card

  cutilDeviceInit(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  cudaSafeCall(cudaMalloc((void **)&d_x, nsize*sizeof(float)));
#if 1
  //h_x2 = (float *)malloc(nsize*sizeof(float));
  cudaSafeCall(cudaMalloc((void **)&d_x2, nsize*sizeof(float)));
#endif
  // execute kernel
  
  /* initiating the value of first vector */
  my_first_kernel<<<nblocks,nthreads>>>(d_x);
  cudaCheckMsg("my_first_kernel execution failed\n");
  
  /* initiating the value of second vector */
  my_second_kernel<<<nblocks,nthreads>>>(d_x2);
  cudaCheckMsg("my_first_kernel execution failed\n");
  
  /* Add the second vector to the first */
  add_vector<<<nblocks,nthreads>>>(d_x2, d_x);
  // copy back results and print them out

  /* copy the result to host vector */
  cudaSafeCall( cudaMemcpy(h_x,d_x2,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

	/* print the result */
  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  cudaSafeCall(cudaFree(d_x));
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
