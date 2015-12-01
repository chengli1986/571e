project(matrixMulCUBLAS)

# set source files
set(PROJECT_SRC matrixMulCUBLAS.cpp)

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

cuda_add_executable(matrixMulCUBLAS ${PROJECT_SRC} OPTIONS ${GENCODE})
cuda_add_cublas_to_target(matrixMulCUBLAS)
install(TARGETS matrixMulCUBLAS
        RUNTIME DESTINATION ${INSTALL_PATH})
