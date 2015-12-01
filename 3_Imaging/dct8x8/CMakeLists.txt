project(dct8x8)

# set source files
set(PROJECT_SRC dct8x8.cu)
set(PROJECT_HEADER dct8x8_kernel_quantization.cuh dct8x8_kernel_short.cuh dct8x8_kernel1.cuh dct8x8_kernel2.cuh)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-G;-g")
set(CUDA_NVCC_FLAGS_RELEASE "-O2")

if (MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "/DEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

cuda_add_executable(dct8x8 ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE})
install(TARGETS dct8x8 
        RUNTIME DESTINATION ${INSTALL_PATH})
