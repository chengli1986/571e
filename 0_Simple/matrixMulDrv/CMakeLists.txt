project(matrixMulDrv)

# set source files
set(PROJECT_SRC matrixMulDrv.cpp)
set(PROJECT_HEADER matrixMul.h)
set(PTX_SRC matrixMul_kernel.cu)

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

if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set(PTX_FILE matrixMul_kernel64.ptx)
else()
    set(PTX_FILE matrixMul_kernel32.ptx)
endif()

cuda_compile_ptx(generated_ptx_files ${PTX_SRC})
cuda_add_executable(matrixMulDrv ${PROJECT_SRC} ${generated_ptx_files})
target_link_libraries(matrixMulDrv ${CUDA_DRIVER_LIBRARY_PATH})
install(TARGETS matrixMulDrv
        RUNTIME DESTINATION ${INSTALL_PATH})
install(FILES ${generated_ptx_files}
        DESTINATION ${INSTALL_PATH}/data
        RENAME ${PTX_FILE})
