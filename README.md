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

* Be careful about the working set size and their corresponding blocks and threads; for example, when working set is 512, i.e. N=512, if only 1 block is used, the thread number has to be set to 512 otherwise there will not be enough threads running the calculation; however, it depends how you write your kernel.

* For atomic operation, sm_11 needs to be added to Makefile so that atomicAdd() function can be recognized.

* Watch out your GPU device shared memory, where in my case, it's only 16KB. Specifying large working set can results in the following error.

```
error:
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
