project(grabcutNPP)

set(PROJECT_SRC GrabcutGMM.cu GrabcutHistogram.cu GrabcutUtil.cu GrabCut.cpp GrabcutMain.cpp)
set(PROJECT_HEADER GPUHistogram.h GrabCut.h)

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

cuda_add_executable(grabcutNPP ${PROJECT_SRC} ${PROJECT_HEADER} OPTIONS ${GENCODE})
target_link_libraries(grabcutNPP nppi nppc freeimage ${ALL_LIBRARIES})
install(TARGETS grabcutNPP
        RUNTIME DESTINATION ${INSTALL_PATH})



