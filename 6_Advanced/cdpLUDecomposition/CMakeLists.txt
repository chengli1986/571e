project(cdpLUDecomposition)

# set source files
set(PROJECT_SRC cdp_lu.cu cdp_lu_main.cu dgetf2.cu dgetrf.cu dlaswp.cu)

# build options
set(GENCODE -gencode=arch=compute_35,code=compute_35;-arch=sm_35)

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

set(CUDA_SEPARABLE_COMPILATION ON)
cuda_add_executable(cdpLUDecomposition ${PROJECT_SRC} OPTIONS "${GENCODE};-lcudadevrt;-lcublas_device")
target_link_libraries(cdpLUDecomposition ${CUDADEVRT_LIBRARIES} cublas_device ${CUDA_CUBLAS_LIBRARIES})
install(TARGETS cdpLUDecomposition 
        RUNTIME DESTINATION ${INSTALL_PATH})
