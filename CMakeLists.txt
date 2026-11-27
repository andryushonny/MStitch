cmake_minimum_required(VERSION 3.24)
project(mstitch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


# No C++ runtime DLLs next to the module: Python on Windows does not look for them on PATH, and a missing
# msvcp140.dll / libstdc++-6.dll gives "DLL load failed while importing _core".
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# zstd: MControl packs snaps as zstd(same-colour difference)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(zstd
  GIT_REPOSITORY https://github.com/facebook/zstd
  GIT_TAG v1.5.6
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR build/cmake)
# pocketfft: header-only FFT for the CPU backend
FetchContent_Declare(pocketfft
  GIT_REPOSITORY https://github.com/mreineck/pocketfft
  GIT_TAG cpp
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(zstd pocketfft)

add_library(mstitch_lib STATIC
  io/frame.cpp
  cpu/fft.cpp
  core/tile.cpp
  core/register.cpp
  core/solve.cpp
  core/render.cpp
  core/session.cpp
  cuda/api.cpp
  cuda/gpu.cpp)
target_include_directories(mstitch_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${pocketfft_SOURCE_DIR}
  ${zstd_SOURCE_DIR}/lib)
target_link_libraries(mstitch_lib PUBLIC libzstd_static)
find_package(Threads REQUIRED)
target_link_libraries(mstitch_lib PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if(MSVC)
  target_compile_options(mstitch_lib PRIVATE /O2 /fp:fast /W3)
else()
  target_compile_options(mstitch_lib PRIVATE -O3 -Wall)
endif()

# CUDA needs nothing at build time: the driver, NVRTC and cuFFT are loaded at run time (cuda/api.cpp).

# Fake CUDA driver + cuFFT on the CPU, for testing the GPU path without a GPU (tests/test_cuda.py)
option(MSTITCH_MOCK_CUDA "Build tests/mock_cuda" OFF)
if(MSTITCH_MOCK_CUDA)
  add_library(mock_cuda SHARED tests/mock_cuda/mock_cuda.cpp)
  target_include_directories(mock_cuda PRIVATE ${pocketfft_SOURCE_DIR})
  set_target_properties(mock_cuda PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

# Python module mstitch._core
if(SKBUILD OR MSTITCH_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core NB_STATIC bindings/module.cpp)
  target_link_libraries(_core PRIVATE mstitch_lib)
  if(MINGW)
    target_link_options(_core PRIVATE -static)
  endif()
  install(TARGETS _core LIBRARY DESTINATION mstitch)
endif()
