fn main() {
    println!("cargo:rustc-link-search=native=llama.cpp/build/bin");
    println!("cargo:rustc-link-lib=dylib=llama");
}

