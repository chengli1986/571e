project(bindlessTexture)

# set source files
set(PROJECT_SRC bindlessTexture.cpp bindlessTexture_kernel.cu)

set(ALL_LIBRARIES ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} ${OPENGL_LIBRARIES})

set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)

cuda_add_executable(bindlessTexture ${PROJECT_SRC} ${PROJECT_HEADER}
                    OPTIONS ${GENCODE} ${DEBUG_OPTION})
target_link_libraries(bindlessTexture ${ALL_LIBRARIES})

install(TARGETS bindlessTexture
        RUNTIME DESTINATION ${INSTALL_PATH})
