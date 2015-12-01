project(quasirandomGenerator)

# set source files
set(PROJECT_SRC quasirandomGenerator.cpp quasirandomGenerator_gold.cpp)
set(PROJECT_CUDA_SM10 quasirandomGenerator_SM10.cu)
set(PROJECT_CUDA_SM13 quasirandomGenerator_SM13.cu)
set(PROJECT_HEADER quasirandomGenerator_common.h realtype.h quasirandomGenerator_kernel.cuh)

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

cuda_add_executable(quasirandomGenerator ${PROJECT_CUDA_SM10} OPTIONS ${GENCODE_SM10})
cuda_add_executable(quasirandomGenerator ${PROJECT_CUDA_SM13} OPTIONS ${GENCODE_SM13})
cuda_add_executable(quasirandomGenerator ${PROJECT_SRC} OPTIONS)

install(TARGETS quasirandomGenerator 
        RUNTIME DESTINATION ${INSTALL_PATH})
