# EECE571E Project

## My mid 2010 MacBook Pro 

Nvidia GeForce GT 330M (Tesla Microarchitecture)

More about this [card](http://www.notebookcheck.net/nvidia-geforce-gt-330m.22437.0.html?id=17654)

Official Product Website: http://www.geforce.com/hardware/notebook-gpus/geforce-gt-330m/features

Wikipedia of Nvidia product line: 
* https://en.wikipedia.org/wiki/GeForce_300_series
* https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units

### GT330M Device Info

```
./bin/x86_64/darwin/release/deviceQuery 

Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GT 330M"
  CUDA Driver Version / Runtime Version          6.0 / 6.0
  CUDA Capability Major/Minor version number:    1.2
  Total amount of global memory:                 256 MBytes (268107776 bytes)
  ( 6) Multiprocessors, (  8) CUDA Cores/MP:     48 CUDA Cores
  GPU Clock rate:                                1100 MHz (1.10 GHz)
  Memory Clock rate:                             790 Mhz
  Memory Bus Width:                              128-bit
  Maximum Texture Dimension Size (x,y,z)         1D=(8192), 2D=(65536, 32768), 3D=(2048, 2048, 2048)
  Maximum Layered 1D Texture Size, (num) layers  1D=(8192), 512 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(8192, 8192), 512 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       16384 bytes
  Total number of registers available per block: 16384
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           512
  Max dimension size of a thread block (x,y,z): (512, 512, 64)
  Max dimension size of a grid size    (x,y,z): (65535, 65535, 1)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             256 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      No
  Device PCI Bus ID / PCI location ID:           1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 6.0, CUDA Runtime Version = 6.0, NumDevs = 1, Device0 = GeForce GT 330M
Result = PASS
```

### GT330M Bandwidth Test Result

```
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: GeForce GT 330M
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			1544.8

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			3306.0

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			21352.8

Result = PASS
```

### Notes

* Be careful about the working set size and their corresponding blocks and threads; for example, when working set is 512, i.e. N=512, if only 1 block is used, the thread number has to be set to 512 otherwise there will not be enough threads running the calculation; however, it depends how you write your kernel. Also, another example is matrix transpose, when N=32 meaning that matrix size is 32 by 32, setting the threads per block to be 32 seems to trample the data easily. Not clear now it's due to shared memory or something else.

* For atomic operation, sm_11 needs to be added to Makefile so that atomicAdd() function can be recognized.

* Watch out your GPU device shared memory, where in my case, it's only 16KB. Specifying large working set can results in the following error.

```
ptxas Error “function uses too much shared data”
```

* excerpt of CUDA SDK simpleAtomicIntrinsics
```
CUDA Sample "simpleAtomics"

This code sample is meant to trivially exercise and demonstrate CUDA's global memory atomic functions:

atomicAdd()
atomicSub()
atomicExch()
atomicMax()
atomicMin()
atomicInc()
atomicdec()
atomicCAS()
atomicAnd()
atomicOr()
atomicXor()

This program requires compute capability 1.1.  To compile the code, therefore, note that the flag "-arch sm_11"
is passed to the nvcc compiler driver in the build step for simpleAtomics.cu.  To use atomics in your programs,
you must pass the same flag to the compiler.

Note that this program is meant to demonstrate the basics of using the atomic instructions, not to demonstrate 
a useful computation.
```

* Typical problems are not friendly multiples of blockDim.x Avoid accessing beyond the end of the arrays:
```
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
        c[index] = a[index] + b[index];
}
```
Update the kernel launch:
```
add<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
```

* Before we look at what atomic operations are and why you care, you should know that atomic opera- tions on global memory are supported only on GPUs of compute capability 1.1 or higher. Furthermore, atomic operations on shared memory require a GPU of compute capability 1.2 or higher. Because of the superset nature of compute capability versions, GPUs of compute capability 1.2 therefore support both shared memory atomics and global memory atomics. Similarly, GPUs of compute capa- bility 1.3 support both of these as well.

* The index of a thread and its thread ID relate to each other in a straightforward way: For a one-dimensional block, they are the same; for a two-dimensional block of size (Dx, Dy),the thread ID of a thread of index (x, y) is (x + y Dx); for a three-dimensional block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy). There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same processor core and must share the limited memory resources of that core. 

* A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp serially executes each branch path taken, disabling threads that are not on that path, and when all paths complete, the threads converge back to the same execution path. Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjoint code paths.

* Each CUDA core is also known as a Streaming Processor or shader unit. The streaming multiprocessor (SM) contains 8 streaming processors (SP). These SMs only get one instruction at time which means that the 8 SPs all execute the same instruction. This is done through a warp ( 32 threads ) where the 8 SPs spend 4 clock cycles executing a single instruction on multiple data (SIMD). Consider the whole GPU to be a couple of SIMD units.... Nvidia calls it SIMT ( Single Instruction Multiple Threads).

* When number of threads is not a multiple of preferred block size, insert bounds test into kernel. Just like what has been done in vectorAdd example where numElements = 50000 and block is set to 96.

* The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently.

* If an atomic instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/ modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined.

* 95% confidence interval: http://www.graphpad.com/guides/prism/6/statistics/index.htm?stat_more_about_confidence_interval.htm

* Using NVCC to generate cubin, PTX, and SASS files
```
/Developer/NVIDIA/CUDA-6.0/bin/nvcc -ptx -gencode arch=compute_12,code=sm_12 vector_add_threads.cu 
/Developer/NVIDIA/CUDA-6.0/bin/nvcc -cubin -gencode arch=compute_12,code=sm_12 vector_add_threads.cu 
/Developer/NVIDIA/CUDA-6.0/bin/cuobjdump -sass vector_add_threads.cubin 
```

Examples are shown below for vector_add_threads.cu

SASS file:
```
dhcp-206-87-194-89:vector_add chwlo$ /Developer/NVIDIA/CUDA-6.0/bin/cuobjdump -sass vector_add_threads.cubin 

	code for sm_12
		Function : _Z11add_threadsPiS_S_
	.headerflags    @"EF_CUDA_SM10 EF_CUDA_PTX_SM(EF_CUDA_SM10)"
        /*0000*/         I2I.U32.U16 R0, R0L;       /* 0x04000780a0000001 */
        /*0008*/         SHL R2, R0, 0x2;           /* 0xc410078030020009 */
        /*0010*/         IADD32 R0, g [0x4], R2;    /* 0x2102e800         */
        /*0014*/         IADD32 R3, g [0x6], R2;    /* 0x2102ec0c         */
        /*0018*/         GLD.U32 R1, global14[R0];  /* 0x80c00780d00e0005 */
        /*0020*/         GLD.U32 R0, global14[R3];  /* 0x80c00780d00e0601 */
        /*0028*/         IADD32 R1, R1, R0;         /* 0x20008204         */
        /*002c*/         IADD32 R0, g [0x8], R2;    /* 0x2102f000         */
        /*0030*/         GST.U32 global14[R0], R1;  /* 0xa0c00781d00e0005 */
		......................................
```
PTX file:
```
	.version 1.4
	.target sm_12, map_f64_to_f32
	// compiled with /Developer/NVIDIA/CUDA-6.0/bin/../open64/lib//be
	// nvopencc 4.1 built on 2014-04-01

	//-----------------------------------------------------------
	// Compiling /var/folders/4p/qx40xw9n1g5bll1hv0m968dm0000gn/T//tmpxft_000032e3_00000000-9_vector_add_threads.cpp3.i (/var/folders/4p/qx40xw9n1g5bll1hv0m968dm0000gn/T/ccBI#.ZvcIe8)
	//-----------------------------------------------------------

	//-----------------------------------------------------------
	// Options:
	//-----------------------------------------------------------
	//  Target:ptx, ISA:sm_12, Endian:little, Pointer Size:64
	//  -O3	(Optimization level)
	//  -g0	(Debug level)
	//  -m2	(Report advisories)
	//-----------------------------------------------------------

	.file	1	"/var/folders/4p/qx40xw9n1g5bll1hv0m968dm0000gn/T//tmpxft_000032e3_00000000-8_vector_add_threads.cudafe2.gpu"
	.file	2	"/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../lib/clang/7.0.0/include/stddef.h"
	.file	3	"/Developer/NVIDIA/CUDA-6.0/bin/../include/crt/device_runtime.h"
	.file	4	"/Developer/NVIDIA/CUDA-6.0/bin/../include/host_defines.h"
	.file	5	"/Developer/NVIDIA/CUDA-6.0/bin/../include/builtin_types.h"
	.file	6	"/Developer/NVIDIA/CUDA-6.0/bin/../include/device_types.h"
	.file	7	"/Developer/NVIDIA/CUDA-6.0/bin/../include/driver_types.h"
	.file	8	"/Developer/NVIDIA/CUDA-6.0/bin/../include/surface_types.h"
	.file	9	"/Developer/NVIDIA/CUDA-6.0/bin/../include/texture_types.h"
	.file	10	"/Developer/NVIDIA/CUDA-6.0/bin/../include/vector_types.h"
	.file	11	"/Developer/NVIDIA/CUDA-6.0/bin/../include/device_launch_parameters.h"
	.file	12	"/Developer/NVIDIA/CUDA-6.0/bin/../include/crt/storage_class.h"
	.file	13	"vector_add_threads.cu"
	.file	14	"/Developer/NVIDIA/CUDA-6.0/bin/../include/common_functions.h"
	.file	15	"/Developer/NVIDIA/CUDA-6.0/bin/../include/math_functions.h"
	.file	16	"/Developer/NVIDIA/CUDA-6.0/bin/../include/math_constants.h"
	.file	17	"/Developer/NVIDIA/CUDA-6.0/bin/../include/device_functions.h"
	.file	18	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_11_atomic_functions.h"
	.file	19	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_12_atomic_functions.h"
	.file	20	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_13_double_functions.h"
	.file	21	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_20_atomic_functions.h"
	.file	22	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_32_atomic_functions.h"
	.file	23	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_35_atomic_functions.h"
	.file	24	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_20_intrinsics.h"
	.file	25	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_30_intrinsics.h"
	.file	26	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_32_intrinsics.h"
	.file	27	"/Developer/NVIDIA/CUDA-6.0/bin/../include/sm_35_intrinsics.h"
	.file	28	"/Developer/NVIDIA/CUDA-6.0/bin/../include/surface_functions.h"
	.file	29	"/Developer/NVIDIA/CUDA-6.0/bin/../include/texture_fetch_functions.h"
	.file	30	"/Developer/NVIDIA/CUDA-6.0/bin/../include/texture_indirect_functions.h"
	.file	31	"/Developer/NVIDIA/CUDA-6.0/bin/../include/surface_indirect_functions.h"
	.file	32	"/Developer/NVIDIA/CUDA-6.0/bin/../include/math_functions_dbl_ptx1.h"


	.entry _Z11add_threadsPiS_S_ (
		.param .u64 __cudaparm__Z11add_threadsPiS_S__a,
		.param .u64 __cudaparm__Z11add_threadsPiS_S__b,
		.param .u64 __cudaparm__Z11add_threadsPiS_S__c)
	{
	.reg .u32 %r<5>;
	.reg .u64 %rd<10>;
	.loc	13	18	0
$LDWbegin__Z11add_threadsPiS_S_:
	.loc	13	20	0
	cvt.u64.u16 	%rd1, %tid.x;
	mul.lo.u64 	%rd2, %rd1, 4;
	ld.param.u64 	%rd3, [__cudaparm__Z11add_threadsPiS_S__a];
	add.u64 	%rd4, %rd3, %rd2;
	ld.global.s32 	%r1, [%rd4+0];
	ld.param.u64 	%rd5, [__cudaparm__Z11add_threadsPiS_S__b];
	add.u64 	%rd6, %rd5, %rd2;
	ld.global.s32 	%r2, [%rd6+0];
	add.s32 	%r3, %r1, %r2;
	ld.param.u64 	%rd7, [__cudaparm__Z11add_threadsPiS_S__c];
	add.u64 	%rd8, %rd7, %rd2;
	st.global.s32 	[%rd8+0], %r3;
	.loc	13	21	0
	exit;
$LDWend__Z11add_threadsPiS_S_:
	} // _Z11add_threadsPiS_S_
```
