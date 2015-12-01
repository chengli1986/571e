project(cdpSimpleQuicksort)

# set source files
set(PROJECT_SRC cdpSimpleQuicksort.cu)

# build options
set(GENCODE -gencode=arch=compute_35,code=compute_35)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

cuda_add_executable(cdpSimpleQuicksort ${PROJECT_SRC} OPTIONS ${GENCODE})
target_link_libraries(cdpSimpleQuicksort ${CUDADEVRT_LIBRARIES})

install(TARGETS cdpSimpleQuicksort 
        RUNTIME DESTINATION ${INSTALL_PATH})
