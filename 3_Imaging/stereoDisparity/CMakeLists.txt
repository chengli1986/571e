project(stereoDisparity)

# set source files
set(PROJECT_SRC stereoDisparity.cu)
set(PROJECT_HEADER stereoDisparity_kernel.cuh)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

cuda_add_executable(stereoDisparity ${PROJECT_SRC} OPTIONS ${GENCODE})
install(TARGETS stereoDisparity 
        RUNTIME DESTINATION ${INSTALL_PATH})
