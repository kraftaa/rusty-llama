### rusty-llama: Rust CLI for LLaMA Model Inference

### Overview
`rusty-llama` is a Rust command-line tool for running Meta’s LLaMA language models locally using the efficient llama.cpp backend. It uses Rust’s FFI to call the native llama.cpp shared library (`libggml.dylib`) for fast, offline text generation on your machine.

### Important Notes
Built and tested on macOS (Apple Silicon arm64). The provided `libggml.dylib` and executable target macOS only.

Linux and Windows require separate builds and compatible shared libraries (WIP).

This project does not include model files due to size and licensing restrictions. You must download and provide your own LLaMA model files (e.g., .gguf or .bin).

The executable loads models at runtime and runs inference locally — no internet required.

### What You Need
Rust executable (`rusty_llama`) compiled for macOS (provided)

Native shared library (`libggml.dylib`) compiled from llama.cpp for macOS (provided)

Pre-trained LLaMA model files (download separately)

### Usage

#### Releases / Download Binaries

Prebuilt binaries and the required libggml.dylib shared library for macOS (Apple Silicon), and Python & Ruby wrapper files are available in the `bin` folder.
```
unzip rusty-llama-release.zip
cd rusty-llama
```
Run all commands from inside that folder for convenience.


#### Downloading the Model
Download model files separately — here’s how to download a quantized LLaMA-2 7B model:

```
# SqueezeNet for image classification
./bin/rusty_llama download-model squeezenet

# LLaMA 2 Chat model for text generation
./bin/rusty_llama download-model llama-2-7b-chat
```

or manually another models
```shell
mkdir -p models
wget -O models/ggml-model-q4_0.gguf \
  https://huggingface.co/TheBloke/LLaMA-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf
```

Place your extra model files into models folder, e.g., `./models/llama-2-7b-chat.Q4_0.gguf`.


#### Set Native Library Path
On macOS, set the environment variable to load the native shared library:

```shell
export DYLD_LIBRARY_PATH=$(pwd)/bin
```


#### REPL Chat Mode

```shell
./bin/rusty_llama --model model_path chat
```

Tip: To get precise answers, format your question clearly, e.g.:
```shell
> Question: What is the day of the week today? Give me the name only.
```

#### Generate from File
```shell
./bin/rusty_llama --model model_path file prompts.txt
```

#### Generate from Prompt
```shell
./bin/rusty_llama --model model_path prompt "Explain Rust ownership rules"
```

#### CSV Query Mode
```shell
./bin/rusty_llama --model model_path csv ./data/sales.csv ./output.txt "Given the following CSV data:\n{csv}\n\nCalculate and output only the numeric average sales. Do not provide explanations or additional text. Answer:"
```
Reads CSV data from `./data/sales.csv`

Inserts CSV contents into the prompt where {csv} is placed

Queries the model and writes output to `./output.txt`

#### Optional Parameters
Customize generation with these flags:

`--temperature <value> `(e.g., 0.5) — randomness of output

-`-top-k <value>` (e.g., 40) — sample from top-k tokens

`--top-p <value>` (e.g., 0.9) — nucleus sampling probability


#### Summarize Text Prompt

Summarize a string of text.

```shell
./bin/rusty_llama --model model_path  summarize-prompt --text "Your long text goes here."
```

If no text is provided, it will warn:

```shell
Prompt can't be empty
```

#### Summarize File
Summarize the contents of a text file:

```shell
./bin/rusty_llama --model model_path summarize-file --filename path/to/file.txt
```

#### Context Answer
Answer a question using an optional context file:

```shell
./bin/rusty_llama --model model_path answer --question "What is Rust?" --context-file path/to/context_file.txt
```
If no context file is provided, the model will answer concisely using general knowledge.


#### Image Classification
Classify an image using an ONNX model:

```shell
./bin/rusty_llama classify --model onnx_model_path/model.onnx --image path_to/image.jpg --labels labels.txt
```
labels.txt contains one label per line, matching the model's output classes.

Prints the top-3 predictions with probabilities.

Example output:

```shell
Top-3 predictions:
  American lobster (0.492)
  horse chestnut seed (0.207)
  Dungeness crab (0.048)
```

Necessary steps:


#### Forecasting
Forecast numeric time series data from a CSV file:

```shell
./bin/rusty_llama forecast --input-data data/forecast_data.csv --steps 5
```

data.csv should contain one numeric value per line (e.g., daily sales, stock prices, quantity).

steps specifies how many future points to predict.

Example output:

```shell
Forecast for next 5 steps:
Step 1: 183.653
Step 2: 189.953
Step 3: 196.253
Step 4: 202.553
Step 5: 208.853
```

#### Example:

```shell
./bin/rusty_llama --model model_path  --temperature 0.5 --top-k 40 --top-p 0.9 chat
```

#### Python and Ruby Wrappers

Quickly interact with your models using these simple scripts:

```shell
python3 ./bin/rusty-llama.py --model ./models/llama-2-7b-chat.Q4_0.gguf chat
```

```shell
ruby ./bin/rusty-llama.rb --model ./models/llama-2-7b-chat.Q4_0.gguf chat
```

#### Makefile Commands
```shell
make download_models — download example models

make run_chat — start chat mode

make run_prompt — run prompt mode with your input
```

### Building from Source (macOS)

Install dependencies:

```shell
xcode-select --install
brew install libomp cmake

```

Clone and build llama.cpp shared library:

```shell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Build the Rust CLI:

```shell
cargo build --release
```

### Summary

- Platform-specific executable and shared library (provided for macOS Apple Silicon )

- Model files must be downloaded separately

- Runs LLaMA models fully offline with local GPU acceleration via Metal

- Flexible CLI and language wrappers make interacting easy

