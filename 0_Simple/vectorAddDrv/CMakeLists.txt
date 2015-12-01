project(vectorAddDrv)

# set source files
set(PROJECT_SRC vectorAddDrv.cpp)
set(PTX_SRC vectorAdd_kernel.cu)

# debug options
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# release options
set(CUDA_NVCC_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set(PTX_FILE vectorAdd_kernel64.ptx)
else()
    set(PTX_FILE vectorAdd_kernel32.ptx)
endif()

cuda_compile_ptx(generated_ptx_files ${PTX_SRC})
cuda_add_executable(vectorAddDrv ${PROJECT_SRC} ${generated_ptx_files})
target_link_libraries(vectorAddDrv ${CUDA_DRIVER_LIBRARY_PATH})
install(TARGETS vectorAddDrv
        RUNTIME DESTINATION ${INSTALL_PATH})
install(FILES ${generated_ptx_files}
        DESTINATION ${INSTALL_PATH}/data
        RENAME ${PTX_FILE})
