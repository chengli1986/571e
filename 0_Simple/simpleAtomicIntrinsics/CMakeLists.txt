project(simpleAtomicIntrinsics)

# set source files
set(PROJECT_SRC simpleAtomicIntrinsics.cu)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_11,code=sm_11)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

cuda_add_executable(simpleAtomicIntrinsics ${PROJECT_SRC} OPTIONS ${GENCODE})
install(TARGETS simpleAtomicIntrinsics 
        RUNTIME DESTINATION ${INSTALL_PATH})
