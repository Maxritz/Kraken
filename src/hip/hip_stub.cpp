// hip_stub.cpp — defines the HIP entry points when this build has no HIP
// support, so the CLI, the tools and the tests link and run as a CPU-only
// reference. The HIP sources are excluded by CMake in that configuration.
#if !defined(KRK_ENABLE_HIP)

#include "krk/backend.hpp"

namespace krk {

int hip_device_count() { return 0; }

std::string hip_runtime_version() { return "unavailable"; }

Backend *make_hip_backend(int device_index, std::string *err) {
    (void)device_index;
    if (err)
        *err = "this binary was built without HIP support (rebuild with "
               "-DKRK_ENABLE_HIP=ON and the HIP SDK on PATH)";
    return nullptr;
}

} // namespace krk

#endif // !KRK_ENABLE_HIP
