project(dwtHaar1D)

# set source files
set(PROJECT_SRC dwtHaar1D.cu)
set(PROJECT_HEADER dwtHaar1D_kernel.cuh)

# build options
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

cuda_add_executable(dwtHaar1D ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE})
install(TARGETS dwtHaar1D 
        RUNTIME DESTINATION ${INSTALL_PATH})
