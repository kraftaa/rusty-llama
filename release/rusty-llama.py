# USAGE
# python3 rusty-llama.py --model models/llama-2-7b-chat.Q4_0.gguf chat

import subprocess
import sys
import argparse
import os
import pathlib

os.environ['DYLD_LIBRARY_PATH'] = str(pathlib.Path(__file__).parent / "bundle")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--top_p", type=float, default=0.9)

subparsers = parser.add_subparsers(dest="mode", required=True)

# Chat mode
subparsers.add_parser("chat")

# Prompt mode
prompt_parser = subparsers.add_parser("prompt")
prompt_parser.add_argument("prompt_text")

# CSV mode
csv_parser = subparsers.add_parser("csv")
csv_parser.add_argument("csv_file")
csv_parser.add_argument("output_file")
csv_parser.add_argument("prompt_text")

args = parser.parse_args()
def run_rusty_llama(model=None, temperature=0.5, top_k=40, top_p=0.9, mode=None, prompt=None, csv_file=None, output_file=None):
    cmd = ["./rusty_llama"]

    # Global options (all modes can accept these)
    cmd += [
        f"--model={model}",
        f"--temperature={temperature}",
        f"--top-k={top_k}",
        f"--top-p={top_p}",
    ]

    # The mode (subcommand)
    cmd.append(mode)

    # Add mode-specific args
    if mode == "chat":
        # no extra args needed for chat mode
        pass
    elif mode == "prompt":
        if prompt:
            cmd.append(prompt)
    elif mode == "csv":
        if csv_file and output_file and prompt:
            cmd += [csv_file, output_file, prompt]
        else:
            raise ValueError("csv mode requires csv_file, output_file, and prompt")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("Running command:", cmd)
    subprocess.run(cmd)

if __name__ == "__main__":

    # mode = args[0]
    args = parser.parse_args()
    print("ARGS", args)
    if not args:
        print("Usage:")
        print("  python3 rusty-llama.py chat ")
        print("  python3 rusty-llama.py prompt <prompt>")
        print("  python3 rusty-llama.py csv <csv_file> <output_file> <prompt>")
        sys.exit(1)

    if args.mode == "prompt":
        run_rusty_llama(
            model=args.model, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            mode=args.mode, prompt=args.prompt_text
        )
    elif args.mode == "chat":
        run_rusty_llama(
            model=args.model, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            mode=args.mode
        )
    elif args.mode == "csv":
        run_rusty_llama(
            model=args.model, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            mode=args.mode, csv_file=args.csv_file, output_file=args.output_file, prompt=args.prompt_text
        )
    else:
        print("Unknown mode:", args.mode)
        sys.exit(1)

