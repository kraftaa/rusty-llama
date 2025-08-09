import subprocess
import sys
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("mode", choices=["chat", "prompt", "csv"])
# parser.add_argument("--prompt", default=None)
# parser.add_argument("--csv_file", default=None)
# parser.add_argument("--output_file", default=None)
# parser.add_argument("--temperature", type=float, default=0.5)
# parser.add_argument("--top_k", type=int, default=40)
# parser.add_argument("--top_p", type=float, default=0.9)

args = parser.parse_args()
def run_rusty_llama(temperature=0.5, top_k=40, top_p=0.9, mode=None, prompt=None, csv_file=None, output_file=None):
    cmd = ["./target/release/rusty_llama"]

    # Global options (all modes can accept these)
    cmd += [
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
    # Simple CLI arg parsing for example:
    # Usage: python3 rusty-llama.py chat "Your prompt here"
    # Or: python3 rusty-llama.py csv ./data.csv ./out.txt "Your prompt for csv"

    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python3 rusty-llama.py chat ")
        print("  python3 rusty-llama.py prompt <prompt>")
        print("  python3 rusty-llama.py csv <csv_file> <output_file> <prompt>")
        sys.exit(1)

    mode = args[0]

    if mode == "prompt":
        prompt = args[1] if len(args) > 1 else None
        run_rusty_llama(mode=mode, prompt=prompt)
    elif mode == "csv":
        if len(args) < 4:
            print("csv mode requires: <csv_file> <output_file> <prompt>")
            sys.exit(1)
        csv_file = args[1]
        output_file = args[2]
        prompt = args[3]
        run_rusty_llama(mode=mode, csv_file=csv_file, output_file=output_file, prompt=prompt)
    elif mode == "chat":
        run_rusty_llama(mode=mode)
    else:
        print("Unknown mode:", mode)
        sys.exit(1)
