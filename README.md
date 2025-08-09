### llama-rs: Rust CLI for LLaMA Model Inference

### Overview

`llama-rs` is a Rust command-line tool that provides local inference of Meta’s LLaMA language model using the efficient llama.cpp backend.

This project uses Rust FFI to call the native llama.cpp shared library (`libggml.dylib`) for fast text generation on the local machine.

### Important Notes on Build and Usage

Built on macOS (Apple Silicon arm64) — the distributed `libggml.dylib` and executable are compiled for macOS only.

Linux or Windows versions require separate builds and compatible shared libraries.

The program does NOT include model files due to their large size and licensing restrictions.

Users should separately download and provide their own LLaMA model files (e.g., .gguf or .bin files).

The executable loads the model file at runtime and uses the shared library for inference.

### What you need to run the program

Rust executable (e.g., `rusty_llama`) — compiled for macOS platform - provided.

Shared library (`libggml.dylib`) — compiled from `llama.cpp` for macOS platform - provided.

LLaMA model files — pre-trained weights, not included in this repo or distribution.
You should download these separately, following official or community sources, and place them in an accessible folder.

### How to use

Download or clone this repo or release archive.


### Downloading the Model

Before running the app, you need to download the model files separately because they are large and subject to licensing.

The default model name is  `llama-2-7b-chat.Q4_0.gguf` (if no model's path provided in command line via `-- --model models/name` )

You can download a standard quantized model like this:


```shell
mkdir -p models
wget -O models/ggml-model-q4_0.gguf \
  https://huggingface.co/TheBloke/LLaMA-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf
```

For chat usage, it’s better to download the chat-optimized model:

```shell
wget -O models/llama-2-7b-chat.gguf \
  https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.gguf
```

Place the model file(s) somewhere, e.g. `./models/llama-2-7b-chat.Q4_0.gguf`

#### Ensure the native library folder is set in your environment

macOS users:

```shell
export DYLD_LIBRARY_PATH=$(pwd)/bundle
```

### REPL chat mode
```shell
./rusty_llama  chat

or 

cargo run -- chat
```
Hint: In order to receive the precise answer, formulate your question like `> Question: What is the day of the week today? Give me the name only.`

### Generate from file

```shell
./rusty_llama file prompts.txt
``` 

### Generate from prompt (multi-word)

```shell
./rusty_llama prompt "Explain Rust ownership rules"
``` 

### CSV query (multi-word query)

```shell
./rusty_llama csv ./data/sales.csv ./output.txt "Given the following CSV data:\n{csv}\n\nCalculate and output only the numeric average sales. Do not provide explanations or additional text. Answer:"

```

#### This command:

Reads CSV data from `./data/sales.csv`.

Inserts CSV content into the prompt where `{csv}` is placed.

Queries the model for an answer.

Saves the generated output to `./output.txt`.

### Optional Parameters

You can customize the text generation behavior by passing optional command-line arguments:

`--temperature <value>` — Controls randomness of the output (e.g., 0.5).

`--top-k <value>` — Limits sampling to the top k tokens (e.g., 40).

`--top-p <value>` — Nucleus sampling probability threshold (e.g., 0.9).

#### Example usage

```shell
./rusty_llama chat --temperature 0.5 --top-k 40 --top-p 0.9
```

### Run the wrapper script

Both Python and Ruby script allow chat, prompt and csv usage.

##### For Python:
```shell

python3 rusty-llama.py --model ./models/llama-2-7b-chat.Q4_0.gguf chat
```

##### For Ruby:

```shell
ruby rusty-llama.rb --model ./models/llama-2-7b-chat.Q4_0.gguf chat
```

### Using Makefile

#### Download models
`make download_models`

#### Run chat mode
`make run_chat`

#### Run prompt mode with your question
`make run_prompt`


### Building from source

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

### Build the Rust CLI:

````shell
cargo build --release
````

### Summary:

The executable and shared library are platform-specific.

The model files must be downloaded and provided by the user separately.

The program dynamically loads the model at runtime and performs inference locally without internet.
