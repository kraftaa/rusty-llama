#!/bin/bash
# Variables
MODEL_DIR = models
MODEL_CHAT = $(MODEL_DIR)/llama-2-7b-chat.Q4_0.gguf
MODEL_BASE = $(MODEL_DIR)/ggml-model-q4_0.gguf
RUSTY_LLAMA = ./rusty_llama

DYLD_LIB_PATH := $(shell pwd)/bundle

.PHONY: all download_models run_chat run_prompt clean

all: download_models
	@echo "You can now run your Rusty Llama binary."

# Download recommended base model
$(MODEL_BASE):
	mkdir -p $(MODEL_DIR)
	wget -O $(MODEL_BASE) https://huggingface.co/TheBloke/LLaMA-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf

# Download recommended chat model
$(MODEL_CHAT):
	mkdir -p $(MODEL_DIR)
	wget -O $(MODEL_CHAT) https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.gguf

download_models: $(MODEL_BASE) $(MODEL_CHAT)

# Run interactive chat mode example
run_chat:
	@DYLD_LIBRARY_PATH=$(DYLD_LIB_PATH) $(RUSTY_LLAMA) --model=$(MODEL_CHAT) chat

# Run prompt mode example (replace with your prompt)
run_prompt:
	@DYLD_LIBRARY_PATH=$(DYLD_LIB_PATH) $(RUSTY_LLAMA) --model=$(MODEL_CHAT) prompt "What is Rust ownership?"

clean:
	rm -rf $(MODEL_DIR)
