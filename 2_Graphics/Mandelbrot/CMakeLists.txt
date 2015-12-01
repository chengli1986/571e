project(Mandelbrot)

# set source files
set(PROJECT_HEADER defines.h tables.h)
set(PROJECT_SRC Mandelbrot.cpp Mandelbrot_kernel.cu)

set(ALL_LIBRARIES ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} ${OPENGL_LIBRARIES})

set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(GENCODE ${GENCODE} -gencode=arch=compute_11,code=sm_11 -gencode=arch=compute_13,code=sm_13)

cuda_add_executable(Mandelbrot ${PROJECT_SRC} ${PROJECT_HEADER}
                    OPTIONS ${GENCODE} ${DEBUG_OPTION})
target_link_libraries(Mandelbrot ${ALL_LIBRARIES})
install(TARGETS Mandelbrot
        RUNTIME DESTINATION ${INSTALL_PATH})
