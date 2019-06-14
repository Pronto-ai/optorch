cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(optorchcpp)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${TORCH_PATH}/lib")

# find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(main optorch/cpp/main.cpp)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
target_compile_definitions(main PRIVATE DISABLE_EASY_PROFILER=1)

target_include_directories(main PUBLIC "${EIGEN3_INCLUDE_DIR}")
target_include_directories(main PUBLIC "${TORCH_PATH}/include")
target_include_directories(main PUBLIC /ceres/ceres-solver/build/config)
target_include_directories(main PUBLIC /ceres/ceres-solver/include)

target_link_libraries(main PUBLIC
"-L${TORCH_PATH}/lib"
-ltorch -lcaffe2 -lc10
/ceres/ceres-solver/build/lib/libceres.a
/usr/lib/x86_64-linux-gnu/libcholmod.a
/usr/lib/lapack/liblapack.a
/usr/lib/libblas/libblas.a
/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
/usr/lib/x86_64-linux-gnu/libcolamd.a
/usr/lib/x86_64-linux-gnu/libcamd.a
/usr/lib/x86_64-linux-gnu/libamd.a
)

find_package(easy_profiler REQUIRED)
target_link_libraries(main PUBLIC easy_profiler)
