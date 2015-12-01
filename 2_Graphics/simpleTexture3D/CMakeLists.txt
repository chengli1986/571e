project(simpleTexture3D)

# set source files
set(PROJECT_SRC simpleTexture3D.cpp simpleTexture3D_kernel.cu)

set(ALL_LIBRARIES ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} ${OPENGL_LIBRARIES})

set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

cuda_add_executable(simpleTexture3D ${PROJECT_SRC} ${PROJECT_HEADER}
                    OPTIONS ${GENCODE} ${DEBUG_OPTION})
target_link_libraries(simpleTexture3D ${ALL_LIBRARIES})
install(TARGETS simpleTexture3D
        RUNTIME DESTINATION ${INSTALL_PATH})
