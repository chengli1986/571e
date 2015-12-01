project(cudaDecodeGL)

# set source files
set(PROJECT_HEADER cudaModuleMgr.h cudaProcessFrame.h FrameQueue.h imageGL.h VideoDecoder.h VideoParser.h VideoSource.h)
set(PROJECT_SRC cudaModuleMgr.cpp cudaProcessFrame.cpp FrameQueue.cpp imageGL.cpp VideoDecoder.cpp VideoParser.cpp VideoSource.cpp videoDecodeGL.cpp NV12ToARGB_drvapi.cu)
set(PTX_SRC NV12ToARGB_drvapi.cu)

set(ALL_LIBRARIES ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} ${OPENGL_LIBRARIES})

# build options
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_11,code=sm_11)

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

if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set(PTX_FILE NV12ToARGB_drvapi64.ptx)
else()
    set(PTX_FILE NV12ToARGB_drvapi32.ptx)
endif()

cuda_compile_ptx(generated_ptx_files ${PTX_SRC})
cuda_add_executable(cudaDecodeGL ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE})
target_link_libraries(cudaDecodeGL ${ALL_LIBRARIES} ${CUDA_nvcuvid_LIBRARY} ${CUDA_DRIVER_LIBRARY_PATH})
install(TARGETS cudaDecodeGL
        RUNTIME DESTINATION ${INSTALL_PATH})
install(FILES ${generated_ptx_files}
        DESTINATION ${INSTALL_PATH}/data
        RENAME ${PTX_FILE})