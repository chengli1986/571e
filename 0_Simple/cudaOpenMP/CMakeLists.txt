project(cudaOpenMP)

# set source files
set(PROJECT_SRC cudaOpenMP.cu)

set(ALL_LIBRARIES ${OPENMP_LIBRARIES})

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

cuda_add_executable(cudaOpenMP ${PROJECT_SRC} OPTIONS ${GENCODE})
target_link_libraries(bindlessTexture ${ALL_LIBRARIES})

install(TARGETS cudaOpenMP 
        RUNTIME DESTINATION ${INSTALL_PATH})
