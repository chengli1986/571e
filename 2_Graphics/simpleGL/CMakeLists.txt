project(simpleGL)

# set source files
set(PROJECT_HEADER defines.h tables.h)
set(PROJECT_SRC simpleGL.cpp simpleGL_kernel.cu)

set(ALL_LIBRARIES ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} ${OPENGL_LIBRARIES})

set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)

cuda_add_executable(simpleGL ${PROJECT_SRC} ${PROJECT_HEADER}
                    OPTIONS ${GENCODE} ${DEBUG_OPTION})
target_link_libraries(simpleGL ${ALL_LIBRARIES})
install(TARGETS simpleGL
        RUNTIME DESTINATION ${INSTALL_PATH})
