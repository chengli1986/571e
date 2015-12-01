project(recursiveGaussian)

# set source files
set(PROJECT_SRC recursiveGaussian.cpp recursiveGaussian_cuda.cu)
set(PROJECT_HEADER recursiveGaussian_kernel.cuh)

set(ALL_LIBRARIES ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY})

set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)

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

# set runtime library for VC
if (MSVC)
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} ";-Xcompiler;/MTd")
  set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} " /MTd")
  
  set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} ";-Xcompiler;/MT")
  set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} " /MT")
endif()

cuda_add_executable(recursiveGaussian ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE} ${DEBUG_OPTION})
target_link_libraries(recursiveGaussian ${ALL_LIBRARIES})
install(TARGETS recursiveGaussian
        RUNTIME DESTINATION ${INSTALL_PATH})
