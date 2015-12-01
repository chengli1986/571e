project(binomialOptions)

# set source files
set(PROJECT_SRC binomialOptions.cpp binomialOptions_gold.cpp)
set(PROJECT_HEADER binomialOptions_common.h realtype.h)
set(PROJECT_CUDA_SM10 binomailOptions_SM10.cu)
set(PROJECT_CUDA_SM13 binomailOptions_SM13.cu)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM13 ${GENCODE} -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_20,code=sm_20)
set(GENCODE_SM10 ${GENCODE} -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_20,code=sm_20)

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

cuda_add_executable(binomialOptions ${PROJECT_CUDA_SM10} OPTIONS ${GENCODE_SM10})
cuda_add_executable(binomialOptions ${PROJECT_CUDA_SM13} OPTIONS ${GENCODE_SM13})
cuda_add_executable(binomialOptions ${PROJECT_SRC} OPTIONS ${GENCODE_SM13})
install(TARGETS binomialOptions 
        RUNTIME DESTINATION ${INSTALL_PATH})
