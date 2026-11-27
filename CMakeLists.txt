cmake_minimum_required(VERSION 3.24)
project(mstitch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MSTITCH_CUDA "Build the CUDA backend (cuda/)" OFF)

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
  core/session.cpp)
target_include_directories(mstitch_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${pocketfft_SOURCE_DIR}
  ${zstd_SOURCE_DIR}/lib)
target_link_libraries(mstitch_lib PUBLIC libzstd_static)
find_package(Threads REQUIRED)
target_link_libraries(mstitch_lib PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(mstitch_lib PRIVATE /O2 /fp:fast /W3)
else()
  target_compile_options(mstitch_lib PRIVATE -O3 -Wall)
endif()

if(MSTITCH_CUDA)
  message(FATAL_ERROR "The CUDA backend is not written yet: see cuda/README.md")
endif()

# Python module mstitch._core
if(SKBUILD OR MSTITCH_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core NB_STATIC bindings/module.cpp)
  target_link_libraries(_core PRIVATE mstitch_lib)
  install(TARGETS _core LIBRARY DESTINATION mstitch)
endif()
