project(matrixMulDynlinkJIT)

# set source files
set(PROJECT_SRC matrixMulDynlinkJIT.cpp)
set(PROJECT_SRC matrixMul_gold.cpp)
set(PROJECT_SRC matrixMul_kernel_32_ptxdump.c)
set(PROJECT_SRC matrixMul_kernel_64_ptxdump.c)
set(PROJECT_SRC cuda_drvapi_dynlink.c)
set(PROJECT_HEADER matrixMul.h)
set(PROJECT_HEADER matrixMul_kernel_32_ptxdump.h)
set(PROJECT_HEADER matrixMul_kernel_64_ptxdump.h)
set(PROJECT_HEADER cuda_drvapi_dynlink.h)
set(PROJECT_HEADER cuda_drvapi_dynlink_cuda.h)
set(PTX_SRC matrixMul_kernel.cu)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set(PTX_FILE matrixMul_kernel64.ptx)
else()
    set(PTX_FILE matrixMul_kernel32.ptx)
endif()

cuda_compile_ptx(generated_ptx_files ${PTX_SRC})
cuda_add_executable(matrixMulDynlinkJIT ${PROJECT_SRC} ${generated_ptx_files})
target_link_libraries(matrixMulDynlinkJIT ${CUDA_DRIVER_LIBRARY_PATH})
install(TARGETS matrixMulDynlinkJIT
        RUNTIME DESTINATION ${INSTALL_PATH})
install(FILES ${generated_ptx_files}
        DESTINATION ${INSTALL_PATH}/data
        RENAME ${PTX_FILE})
