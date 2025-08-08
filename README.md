**llama-rs: Rust CLI for LLaMA Model Inference**

**Overview**

`llama-rs` is a Rust command-line tool that provides local inference of Meta’s LLaMA language model using the efficient llama.cpp backend.

This project uses Rust FFI to call the native llama.cpp shared library (`libggml.dylib`) for fast text generation on the local machine.

**Important Notes on Build and Usage**

Built on macOS (Apple Silicon arm64) — the distributed `libggml.dylib` and executable are compiled for macOS only.

Linux or Windows versions require separate builds and compatible shared libraries.

The program does NOT include model files due to their large size and licensing restrictions.

Users should separately download and provide their own LLaMA model files (e.g., .gguf or .bin files).

The executable loads the model file at runtime and uses the shared library for inference.

What you need to run the program

Rust executable (e.g., `llama_ffi_demo`) — compiled for your platform.

Shared library (`libggml.dylib`) — compiled from `llama.cpp` for your platform.

LLaMA model files — pre-trained weights, not included in this repo or distribution.
You must download these separately, following official or community sources, and place them in an accessible folder.

**How to use**

The default model (if not models path provided in command line : `llama-2-7b-chat.Q4_0.gguf` )
Assuming you have the model in `models/...` required files in the same directory, you can run:

```
./llama_ffi_demo csv ./data/sales.csv ./output.txt "Given the following CSV data:\n{csv}\n\nCalculate the average sales for all customers.\nAnswer ONLY with the numeric average."
```

This command:

Reads CSV data from `./data/sales.csv`.

Inserts CSV content into the prompt where `{csv}` is placed.

Queries the model for an answer.

Saves the generated output to `./output.txt`.

Building from source

If you want to build the executable yourself on macOS:

Clone and build the llama.cpp shared library:

```shell
xcode-select --install
brew install libomp
brew install cmake

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

**Build the Rust CLI:**

````shell
cargo build --release
````

**Summary**
The executable and shared library are platform-specific.

The model files must be downloaded and provided by the user separately.

The program dynamically loads the model at runtime and performs inference locally without internet.

License
MIT License. See LICENSE for details.