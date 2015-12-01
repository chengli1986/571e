project(HSOpticalFlow)

# set source files
set(PROJECT_SRC main.cpp flowGold.cpp flowCUDA.cu)
set(PROJECT_HEADER flowCUDA.h flowGold.h common.h addKernel.cuh derivativesKernel.cuh downscaleKernel.cuh solverKernel.cuh solverKernel.cuh upscaleKernel.cuh warpingKernel.cuh)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

cuda_add_executable(HSOpticalFlow ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE})
install(TARGETS HSOpticalFlow 
        RUNTIME DESTINATION ${INSTALL_PATH})
