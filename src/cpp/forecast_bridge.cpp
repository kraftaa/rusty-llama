#include <vector>
#include <cstring>

extern "C" {

int forecast_init(const char* model_path) {
    (void)model_path;
    // Real impl: load torch::jit::script::Module here.
    return 1;
}

int forecast_run(const float* values_ptr, int len, int steps, float* out_ptr) {
    if (!values_ptr || !out_ptr || len <= 0 || steps <= 0) return 0;
    // Stub: naive continuation = repeat last value
    float last = values_ptr[len-1];
    for (int i = 0; i < steps; ++i) out_ptr[i] = last;
    return 1;
}

} // extern "C"

