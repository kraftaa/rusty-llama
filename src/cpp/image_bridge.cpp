#include <string>
#include <vector>
#include <cstring>

// In a real impl, include onnxruntime_cxx_api.h and load the model.
// We'll stub minimal behavior so the pipeline compiles.

static std::string g_label = "uninitialized";

extern "C" {

int image_init(const char* onnx_path, const char* labels_path) {
    (void)onnx_path;
    // Load first label from labels file as a sanity check
    FILE* f = fopen(labels_path, "r");
    if (!f) return 0;
    char buf[256];
    if (fgets(buf, sizeof(buf), f) != nullptr) {
        // trim newline
        size_t n = strlen(buf);
        while (n && (buf[n-1]=='\n' || buf[n-1]=='\r')) { buf[--n] = 0; }
        g_label = std::string(buf);
    } else {
        g_label = "unknown";
    }
    fclose(f);
    return 1;
}

const char* image_classify(const char* image_path) {
    // Stub: pretend we ran inference and return the first label + filename
    static std::string out;
    out = g_label + " (stubbed) <- " + std::string(image_path);
    return out.c_str();
}

} // extern "C"

