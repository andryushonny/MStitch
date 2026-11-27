// CUDA without the CUDA SDK at build time: the driver (nvcuda / libcuda), NVRTC and cuFFT are loaded at run time,
// so one build runs everywhere and uses the GPU when the libraries are there. Only what MStitch calls is declared.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace ms::cuda {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st; using CUcontext = CUctx_st*;
struct CUmod_st; using CUmodule = CUmod_st*;
struct CUfunc_st; using CUfunction = CUfunc_st*;
struct CUstream_st; using CUstream = CUstream_st*;
using nvrtcResult = int;
struct _nvrtcProgram; using nvrtcProgram = _nvrtcProgram*;
using cufftHandle = int;
using cufftResult = int;

constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
constexpr int CUFFT_R2C = 0x2a;
constexpr int CUFFT_C2R = 0x2c;

struct Api {
    // driver
    CUresult (*cuInit)(unsigned) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;
    CUresult (*cuMemGetInfo)(size_t*, size_t*) = nullptr;           // _v2
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;         // _v2
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;                   // _v2
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;   // _v2
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;         // _v2
    CUresult (*cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t) = nullptr;   // _v2
    CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                               CUstream, void**, void**) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
    // NVRTC
    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*,
                                      const char* const*) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
    nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
    nvrtcResult (*nvrtcVersion)(int*, int*) = nullptr;
    // cuFFT
    cufftResult (*cufftPlanMany)(cufftHandle*, int, int*, int*, int, int, int*, int, int, int, int) = nullptr;
    cufftResult (*cufftExecR2C)(cufftHandle, float*, void*) = nullptr;
    cufftResult (*cufftExecC2R)(cufftHandle, void*, float*) = nullptr;
    cufftResult (*cufftDestroy)(cufftHandle) = nullptr;
    cufftResult (*cufftGetVersion)(int*) = nullptr;

    bool has_driver = false, has_nvrtc = false, has_cufft = false;
};

// Load the libraries from the first path that opens in each list (bare names use the system search).
// Returns a line per library saying what was found; safe to call again (keeps what is loaded).
std::string load(const std::vector<std::string>& driver, const std::vector<std::string>& nvrtc,
                 const std::vector<std::string>& cufft);
Api& api();

}  // namespace ms::cuda
