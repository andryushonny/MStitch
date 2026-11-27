#include "cuda/api.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ms::cuda {

namespace {
void* open_lib(const std::string& p) {
#ifdef _WIN32
    return (void*)LoadLibraryExA(p.c_str(), nullptr,
                                 p.find_first_of("/\\") != std::string::npos ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
#else
    return dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}
void* sym(void* h, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)h, name);
#else
    return dlsym(h, name);
#endif
}
template <class F>
bool get(void* h, F& f, const char* name) {
    f = reinterpret_cast<F>(sym(h, name));
    return f != nullptr;
}
void* first(const std::vector<std::string>& paths, std::string& which) {
    for (auto& p : paths)
        if (void* h = open_lib(p)) { which = p; return h; }
    return nullptr;
}
}  // namespace

Api& api() {
    static Api a;
    return a;
}

std::string load(const std::vector<std::string>& driver, const std::vector<std::string>& nvrtc,
                 const std::vector<std::string>& cufft) {
    Api& a = api();
    std::string out, which;
    if (!a.has_driver) {
        if (void* h = first(driver, which)) {
            bool ok = get(h, a.cuInit, "cuInit") && get(h, a.cuDeviceGetCount, "cuDeviceGetCount") &&
                      get(h, a.cuDeviceGet, "cuDeviceGet") && get(h, a.cuDeviceGetName, "cuDeviceGetName") &&
                      get(h, a.cuDeviceGetAttribute, "cuDeviceGetAttribute") &&
                      get(h, a.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
                      get(h, a.cuCtxSetCurrent, "cuCtxSetCurrent") && get(h, a.cuCtxSynchronize, "cuCtxSynchronize") &&
                      get(h, a.cuMemGetInfo, "cuMemGetInfo_v2") && get(h, a.cuMemAlloc, "cuMemAlloc_v2") &&
                      get(h, a.cuMemFree, "cuMemFree_v2") && get(h, a.cuMemcpyHtoD, "cuMemcpyHtoD_v2") &&
                      get(h, a.cuMemcpyDtoH, "cuMemcpyDtoH_v2") && get(h, a.cuMemcpyDtoD, "cuMemcpyDtoD_v2") &&
                      get(h, a.cuModuleLoadData, "cuModuleLoadData") &&
                      get(h, a.cuModuleGetFunction, "cuModuleGetFunction") &&
                      get(h, a.cuLaunchKernel, "cuLaunchKernel") && get(h, a.cuGetErrorString, "cuGetErrorString");
            a.has_driver = ok;
            out += ok ? "driver: " + which + "\n" : "driver: " + which + " lacks functions\n";
        } else {
            out += "driver: not found (no NVIDIA driver)\n";
        }
    }
    if (!a.has_nvrtc) {
        if (void* h = first(nvrtc, which)) {
            bool ok = get(h, a.nvrtcCreateProgram, "nvrtcCreateProgram") &&
                      get(h, a.nvrtcCompileProgram, "nvrtcCompileProgram") &&
                      get(h, a.nvrtcGetCUBINSize, "nvrtcGetCUBINSize") && get(h, a.nvrtcGetCUBIN, "nvrtcGetCUBIN") &&
                      get(h, a.nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize") &&
                      get(h, a.nvrtcGetProgramLog, "nvrtcGetProgramLog") &&
                      get(h, a.nvrtcDestroyProgram, "nvrtcDestroyProgram") && get(h, a.nvrtcVersion, "nvrtcVersion");
            a.has_nvrtc = ok;
            out += ok ? "nvrtc: " + which + "\n" : "nvrtc: " + which + " lacks functions\n";
        } else {
            out += "nvrtc: not found (pip install nvidia-cuda-nvrtc-cu12)\n";
        }
    }
    if (!a.has_cufft) {
        if (void* h = first(cufft, which)) {
            bool ok = get(h, a.cufftPlanMany, "cufftPlanMany") && get(h, a.cufftExecR2C, "cufftExecR2C") &&
                      get(h, a.cufftExecC2R, "cufftExecC2R") && get(h, a.cufftDestroy, "cufftDestroy") &&
                      get(h, a.cufftGetVersion, "cufftGetVersion");
            a.has_cufft = ok;
            out += ok ? "cufft: " + which + "\n" : "cufft: " + which + " lacks functions\n";
        } else {
            out += "cufft: not found (pip install nvidia-cufft-cu12)\n";
        }
    }
    return out;
}

}  // namespace ms::cuda
