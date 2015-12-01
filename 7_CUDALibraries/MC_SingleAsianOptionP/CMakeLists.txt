project(MC_SingleAsianOptionP)

# set source files
set(PROJECT_SRC src/pricingengine.cu src/main.cpp src/test.cpp)
set(PROJECT_HEADER inc/cudasharedmem.h inc/test.h inc/asianoption.h inc/pricingengine.h)

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_13,code=sm_13)

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

cuda_add_executable(MC_SingleAsianOptionP ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE})
target_link_libraries(MC_SingleAsianOptionP ${CUDA_curand_LIBRARY})

install(TARGETS MC_SingleAsianOptionP
        RUNTIME DESTINATION ${INSTALL_PATH})
