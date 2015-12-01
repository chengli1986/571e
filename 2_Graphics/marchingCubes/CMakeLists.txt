project(marchingCubes)

# set source files
set(PROJECT_HEADER defines.h tables.h)
set(PROJECT_SRC marchingCubes.cpp marchingCubes_kernel.cu)

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

set(ALL_LIBRARIES ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} ${OPENGL_LIBRARIES})

set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

cuda_add_executable(marchingCubes ${PROJECT_SRC} ${PROJECT_HEADER}
                    OPTIONS ${GENCODE} ${DEBUG_OPTION})
target_link_libraries(marchingCubes ${ALL_LIBRARIES})
install(TARGETS marchingCubes
        RUNTIME DESTINATION ${INSTALL_PATH})
