project(convolutionFFT2D)

# set source files
set(PROJECT_SRC convolutionFFT2D.cu)
set(PROJECT_HEADER convolution2D_common.h convolutionFFT2D.cuh)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

cuda_add_executable(convolutionFFT2D ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE})
target_link_libraries(convolutionFFT2D ${CUFFT_LIBRARY})

install(TARGETS convolutionFFT2D
        RUNTIME DESTINATION ${INSTALL_PATH})
