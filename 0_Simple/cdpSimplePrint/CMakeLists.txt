project(cdpSimplePrint)

# set source files
set(PROJECT_SRC cdpSimplePrint.cu)

# build options
set(GENCODE -gencode=arch=compute_35,code=sm_35)

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

set(CUDA_SEPARABLE_COMPILATION ON)
cuda_compile(DEVICE_OBJS ${PROJECT_SRC} OPTIONS ${GENCODE})
cuda_compute_separable_compilation_object_file_name(LINK_OBJS cdpSimplePrint ${DEVICE_OBJS})
cuda_link_separable_compilation_objects(${LINK_OBJS} cdpSimplePrint ${GENCODE} ${DEVICE_OBJS} ${CUDADEVRT_LIBRARIES})

add_executable(cdpSimplePrint 
               ${DEVICE_OBJS}
               ${LINK_OBJS})
            
target_link_libraries(cdpSimplePrint ${CUDADEVRT_LIBRARIES})

set_target_properties(cdpSimplePrint 
                      PROPERTIES
                      LINKER_LANGUAGE ${CUDA_C_OR_CXX})

install(TARGETS cdpSimplePrint 
        RUNTIME DESTINATION ${INSTALL_PATH})
