//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//////////////////#include <cutil_inline.h>

//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x;
}


//
// main code
//

int main(int argc, char **argv)
{
  /* host copy of h_x */
  float *h_x;
  /* device copy of d_x */
  float *d_x;
  int   nblocks, nthreads, nsize; 

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array
  // reserve memory on HOST
  h_x = (float *)malloc(nsize*sizeof(float));
  // reserve memory on Device
  cudaMalloc((void **)&d_x, nsize*sizeof(float));

  // execute kernel

  my_first_kernel<<<nblocks,nthreads>>>(d_x);

  // copy back results and print them out

  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);

  for (int n=0; n<nsize; n++) 
     printf("INFO: thread count,  thread ID  =  %d  %f \n",n,h_x[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
