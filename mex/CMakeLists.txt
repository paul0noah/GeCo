if(OpenMP_CXX_FOUND)
    message("Compiling with OPENMP")
    if(WITH_CUDA)
        matlab_add_mex( NAME ${PROJECT_NAME}
                            SRC surface_cycles.cpp
                            LINK_TO igl::core tsl::robin_map Thrust OpenMP::OpenMP_CXX CUDA::cudart)
    else()
        matlab_add_mex( NAME ${PROJECT_NAME}
                            SRC surface_cycles.cpp
                            LINK_TO igl::core tsl::robin_map Thrust OpenMP::OpenMP_CXX)
    endif()
else()
    message("Compiling without OPENMP")
    if(WITH_CUDA)
        matlab_add_mex( NAME ${PROJECT_NAME}
                            SRC surface_cycles.cpp
                            LINK_TO igl::core tsl::robin_map Thrust CUDA::cudart)
    else()
        matlab_add_mex( NAME ${PROJECT_NAME}
                            SRC surface_cycles.cpp
                            LINK_TO igl::core tsl::robin_map Thrust)
    endif()
endif()