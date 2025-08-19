fn main() {
    // Tell Rust where to find native dylibs
    println!("cargo:rustc-link-search=native=bin");

    // Link llama.cpp library
    println!("cargo:rustc-link-lib=dylib=llama");

    // Link ONNX Runtime library
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    // Add rpath so the binary looks for dylibs relative to itself (macOS)
    // println!("cargo:rustc-link-arg=-rpath,@executable_path/bin");
    // println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/bin");
}

