use std::ffi::{CString, CStr};
use std::os::raw::{c_int, c_float, c_char, c_void};
use std::io::{self, Write};
use std::fs::File;
use std::ptr;
use clap::Subcommand;
type LlamaToken = i32; // check actual type in the headers
type LlamaPos = i32;   // adjust if different
type LlamaSeqId = i32; // adjust if different


#[repr(C)]
#[derive(Debug)]
pub struct LlamaModel;
#[repr(C)]
pub struct LlamaContext;

#[repr(C)]
pub struct LlamaBatch {
    pub n_tokens: c_int,
    pub token: *mut LlamaToken,
    pub embd: *mut c_float,
    pub pos: *mut LlamaPos,
    pub n_seq_id: *mut c_int,
    pub seq_id: *mut *mut LlamaSeqId,
    pub logits: *mut i8,
}
#[repr(C)]
pub struct LlamaModelParams {
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub tensor_split_size: usize,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub progress_callback: Option<extern "C" fn(f32, *mut std::ffi::c_void)>,
    pub progress_callback_user_data: *mut std::ffi::c_void,
}

#[repr(C)]
pub struct LlamaContextParams {
    pub seed: u32,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: i32,
    pub mul_mat_q: bool,
    pub f16_kv: bool,
    pub logits_all: bool,
    pub embedding: bool,
    pub offload_kqv: bool,
    pub offload_kqv_linear: bool,
    pub flash_attn: bool,
    pub pooling_type: i32,
    pub rms_norm_eps: f32,
}

#[repr(C)]
pub struct LlamaVocab;  // Opaque pointer type

#[repr(C)]
pub struct LlamaTokenDataArray {
    pub data: *mut LlamaTokenData,
    pub size: usize,
    pub sorted: bool,
}

#[repr(C)]
#[derive(Debug)]
pub struct LlamaSampler;  // Opaque pointer type

#[repr(C)]
pub struct LlamaSamplerI {
    // Function pointers defining behavior (e.g., sample_token, etc.)
    // Leave empty or mock if using the built-in samplers.
}
#[repr(C)]
pub struct SamplerParams;  // Opaque pointer type

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LlamaTokenData {
    pub id: i32,     // llama_token is typedef'd to int32_t
    pub logit: f32,
    pub p: f32,
}

#[repr(C)]
struct LlamaMemory {
    _private: [u8; 0], // opaque struct
}

#[link(name = "llama")]
extern "C" {
    fn llama_backend_init();
    fn llama_model_default_params() -> LlamaModelParams;
    fn llama_context_default_params() -> LlamaContextParams;
    fn llama_load_model_from_file(path: *const c_char, params: LlamaModelParams) -> *mut LlamaModel;
    fn llama_new_context_with_model(model: *mut LlamaModel, params: LlamaContextParams) -> *mut LlamaContext;
    fn llama_free(ctx: *mut LlamaContext);
    fn llama_free_model(model: *mut LlamaModel);

    // Instead of llama_eval
    fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> c_int;

    fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    fn llama_vocab_get_text(vocab: *const LlamaVocab, token: i32) -> *const c_char;

    fn llama_sampler_init(
        iface: *const LlamaSamplerI,
        ctx: *mut LlamaContext,
    ) -> *mut LlamaSampler;
    fn llama_tokenize(
        vocab: *const LlamaVocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut i32,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    // Sampler functions (adjust names)
    fn llama_sampler_free(sampler: *mut LlamaSampler);
    // pub fn llama_sampler_sample(sampler: *mut LlamaSampler, ctx: *mut LlamaContext, idx: i32) -> i32;
    pub fn llama_sampler_sample(
        sampler: *mut LlamaSampler,
        ctx: *mut LlamaContext,
        idx: i32,
    ) -> LlamaToken;
    pub fn llama_sampler_accept(sampler: *mut LlamaSampler, token: i32);
    pub fn llama_sampler_apply(sampler: *mut LlamaSampler, logits: *mut LlamaTokenDataArray);

    fn llama_sampler_init_greedy() -> *mut LlamaSampler;
    fn llama_sample(sampler: *mut LlamaSampler, ctx: *mut LlamaContext);
    fn llama_sampler_accept_token(sampler: *mut LlamaSampler, ctx: *mut LlamaContext) -> i32;

    fn llama_get_logits(ctx: *mut LlamaContext) -> *const f32;
    // fn llama_n_vocab(ctx: *const LlamaContext) -> i32;
    fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> i32;

    // fn llama_sample_init_greedy(ctx: *mut LlamaContext, logits: *const f32) -> i32;

    // Instead of llama_token_to_str
    fn llama_token_get_text(ctx: *const LlamaContext, token: i32) -> *const c_char;

    fn llama_sampler_create(params: *const SamplerParams) -> *mut LlamaSampler;
    // pub fn llama_get_memory(ctx: *mut LlamaContext) -> *mut llama_memory;
    // DEPRECATED(LLAMA_API struct llama_kv_cache * llama_get_kv_self(struct llama_context * ctx), "use llama_get_memory instead");
    // pub fn llama_get_memory(ctx: *const LlamaContext) -> llama_memory_t;
    fn llama_get_memory(ctx: *const LlamaContext) -> *mut LlamaMemory;
    // LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
    // pub type llama_memory_t;
    // Clear the llama memory; data=true clears buffers, false clears only metadata
    fn llama_memory_clear(mem: *mut LlamaMemory, data: bool);
    // LLAMA_API void llama_memory_clear(
    // llama_memory_t mem,
    // bool data);
    // fn llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32) -> *mut LlamaSampler;
    pub fn llama_sampler_init_temp(t: f32) -> *mut LlamaSampler;
    pub fn llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32) -> *mut LlamaSampler;

    // LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);


}

use clap::{Parser};

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive chat mode
    Chat,
    /// Generate from a file
    File {
        filename: String,
    },
    /// Generate from a prompt passed directly
    Prompt {
        text: Vec<String>,
    },
    /// Query a CSV file
    Csv {
        csv_path: String,
        output_path: String,
        query: Vec<String>,
    },
}

// Clears previous model context memory to start fresh
//
// Tokenizes the prompt into model tokens
//
// Feeds those tokens into the model (decoding the prompt)
//
// Generates new tokens step-by-step (sampling tokens with the sampler)
//
// Converts tokens back to text and collects the generated output string
//
// Stops on EOS/BOS tokens or max tokens reached
unsafe fn generate_text(
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    sampler: *mut LlamaSampler,
    prompt: &str,
) -> String {

    // Reset the model's memory to clear past context
    let mem = llama_get_memory(ctx);
    assert!(!mem.is_null(), "Memory pointer is null");
    llama_memory_clear(mem, true);

    let full_prompt = with_instruction(&prompt);
    // Tokenize the prompt string into model tokens
    // let prompt_c = CString::new(prompt).unwrap();
    let prompt_c = CString::new(full_prompt.clone()).unwrap();
    let mut tokens = [0i32; 512];
    let n_tokens = llama_tokenize(
        vocab,
        prompt_c.as_ptr(),
        full_prompt.len() as i32,
        // prompt.len() as i32,
        // prompt.len() as i32,
        tokens.as_mut_ptr(),
        tokens.len() as i32,
        true,  // add_special tokens
        false, // parse_special tokens
    );
    assert!(n_tokens > 0, "Tokenization failed");
    // println!("Prompt sent to model:\n{}", full_prompt);
    println!("Token count: {}", n_tokens);
    // Feed the entire prompt tokens at once with consecutive positions
    let pos: Vec<i32> = (0..n_tokens).collect();
    let mut batch = LlamaBatch {
        n_tokens,
        token: tokens.as_mut_ptr(),
        embd: ptr::null_mut(),
        pos: pos.as_ptr() as *mut i32,
        n_seq_id: ptr::null_mut(),
        seq_id: ptr::null_mut(),
        logits: ptr::null_mut(),
    };
    let ret = llama_decode(ctx, batch);
    assert_eq!(ret, 0, "Decode failed on prompt tokens");

    // Prepare to generate tokens for completion
    let max_tokens = 250;

    let mut cur_len = n_tokens as usize;
    print!("→ ");
    io::stdout().flush().unwrap();

    let mut n_past = n_tokens;
    let mut generated_tokens = Vec::new();
    let mut output = String::new();
    let mut newline_count = 0;

    for _ in 0..max_tokens {
        let next_token = generated_tokens.last().unwrap_or(&tokens[n_past as usize - 1]);
        // implement breaking after 3 new lines
        if next_token == &13 {  // newline
            newline_count += 1;
            if newline_count >= 3 {
                break; // stop after 3 consecutive newlines
            }
        } else {
            newline_count = 0;  // reset count on other tokens
        }

        let mut token_slice = [*next_token];
        let mut pos = [n_past];

        let mut logits_required = [1i8];

        let mut batch = LlamaBatch {
            n_tokens: 1,
            token: token_slice.as_mut_ptr(),
            embd: ptr::null_mut(),
            pos: pos.as_ptr() as *mut i32,
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: logits_required.as_ptr() as *mut i8,
        };

        let ret = llama_decode(ctx, batch);
        assert_eq!(ret, 0, "Decode failed");

        let logits_ptr = llama_get_logits(ctx);
        assert!(!logits_ptr.is_null(), "Logits pointer is null");

        let vocab_size = llama_vocab_n_tokens(vocab);

        // Wrap logits into token data for sampler
        let mut token_data: Vec<LlamaTokenData> = (0..vocab_size).map(|i| {
            LlamaTokenData {
                id: i as i32,
                logit: *logits_ptr.add(i as usize),
                p: 0.0,
            }
        }).collect();

        let mut token_data_array = LlamaTokenDataArray {
            data: token_data.as_mut_ptr(),
            size: token_data.len(),
            sorted: false,
        };

        // Apply sampling to modify logits (e.g., greedy, temperature, top-p)
        llama_sampler_apply(sampler, &mut token_data_array);

        // Sample next token index from sampler
        let next_token = llama_sampler_sample(sampler, ctx, 0);

        llama_sampler_accept(sampler, next_token);

        if cur_len < tokens.len() {
            tokens[cur_len] = next_token;
            cur_len += 1;
        } else {
            println!("\n[Token buffer full]");
            break;
        }

        let token_text_ptr = llama_vocab_get_text(vocab, next_token);

        let mut token_text = std::ffi::CStr::from_ptr(token_text_ptr)
            .to_string_lossy()
            .into_owned();

        // Replace visible hex newlines with real ones and clean up whitespace token
        token_text = token_text.replace("<0x0A>", "\n");
        token_text = token_text.replace('▁', " ");
        print!("{}", token_text);
        io::stdout().flush().unwrap();
        output.push_str(&token_text);
        // println!("NEXT_TOKEN {:?}", &next_token );
        // println!("NEXT_TOKEN text {:?}", &token_text );
        // Stop generation if EOS or BOS tokens appear -- doesn't work now as often have </s> token
        // if next_token == 2 || next_token == 0 {
        //     break;
        // }
        // if next_token == 2 {
        //     break;  // Stop generating tokens after token 2 - but it never happens
        // }
        n_past += 1;
        generated_tokens.push(next_token);
    }
    println!();
    output
}

use csv::Reader;
use std::fs;

fn read_csv_file(filepath: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut rdr = Reader::from_path(filepath)?;
    let mut csv_content = String::new();

    // Convert csv rows to string, e.g. tabular text
    for result in rdr.records() {
        let record = result?;
        csv_content.push_str(&record.iter().collect::<Vec<_>>().join(", "));
        csv_content.push('\n');
    }

    Ok(csv_content)
}

fn read_csv_with_header(csv_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(csv_path)?;

    // Check if the first line looks like a header (contains alphabetic characters)
    let first_line = content.lines().next().unwrap_or("");
    let has_header = first_line.chars().any(|c| c.is_alphabetic());

    let full_csv = if has_header {
        content
    } else {
        println!("No headers in the file");
        "No headers in the file".to_string()
        // Prepend your known header line if missing
        // let header = "customer_name,sales\n";
        // format!("{}{}", header, content)
    };

    Ok(full_csv)
}
unsafe fn run_csv_query(
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    sampler: *mut LlamaSampler,
    csv_path: &str,
    // query: &str,
    prompt_template: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // let csv_text = read_csv_file(csv_path)?;
    let csv_text = read_csv_with_header(csv_path)?;
    println!("CSV content:\n{}", csv_text);
    // Create prompt combining csv data + question
    // let prompt = format!(
    //     "Given the following CSV data:\n{}\n\n{}",
    //     csv_text, query
    // );


    let prompt_template = prompt_template.replace("\\n", "\n");
    // Replace placeholder {csv} in prompt template with csv_text
    let prompt = prompt_template.replace("{csv}", &csv_text);
    println!("Final prompt sent to model:\n{}", prompt);
    let generated = generate_text(ctx, vocab, sampler, &prompt);

    // Save generated output to file
    let mut file = File::create(output_path)?;
    file.write_all(generated.as_bytes())?;

    println!("Generated output saved to {}", output_path);

    Ok(())
}

fn with_instruction(prompt: &str) -> String {
    format!(
            // "Answer ONLY the question below concisely:\n\n{}",
            "Provide ONLY the direct answer below. DO NOT repeat the question or add explanations.\n\n{}",
        // "Provide ONLY the numeric answer below, as an integer, with no text, punctuation, or explanation.\n\n{}",
        prompt)
}

fn main() {
    unsafe {
        llama_backend_init();

        let model_path = CString::new("models/llama-2-7b-chat.Q4_0.gguf").unwrap();
        let model_params = llama_model_default_params();
        let model = llama_load_model_from_file(model_path.as_ptr(), model_params);
        assert!(!model.is_null(), "Failed to load model");

        let ctx_params = llama_context_default_params();
        let ctx = llama_new_context_with_model(model, ctx_params);
        assert!(!ctx.is_null(), "Failed to create context");

        let vocab = llama_model_get_vocab(model);
        assert!(!vocab.is_null(), "Failed to get vocab");

        let sampler = llama_sampler_init_greedy();
        assert!(!sampler.is_null(), "Failed to init sampler");

        let cli = Cli::parse();
        match cli.command {
            Commands::Chat => {
                // REPL chat mode
                loop {
                    print!("\n> ");
                    io::stdout().flush().unwrap();

                    let mut input = String::new();
                    io::stdin().read_line(&mut input).unwrap();
                    let input = input.trim();

                    if input == "exit" {
                        break;
                    }

                    generate_text(ctx, vocab, sampler, input);
                }
            }
            Commands::File { filename } => {
                let text = fs::read_to_string(filename).unwrap();
                generate_text(ctx, vocab, sampler, &text);
            }
            Commands::Prompt { text } => {
                let prompt = text.join(" ");
                generate_text(ctx, vocab, sampler, &prompt);
            }
            Commands::Csv {
                csv_path,
                output_path,
                query,
            } => {
                let prompt_template = query.join(" ").replace("\\n", "\n");

                run_csv_query(ctx, vocab, sampler, &csv_path, &prompt_template, &output_path).unwrap();
            }
        }

        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_free_model(model);
    }
}


// # REPL chat mode
// cargo run -- chat
//
// # Generate from file
// cargo run -- file prompts.txt
//
// # Generate from prompt (multi-word)
// cargo run -- prompt Explain Rust ownership rules
//
// # CSV query (multi-word query)
// cargo run -- csv ./data/sales.csv ./output.txt List all unique customers

// main idea
// fn main() {
//     unsafe {
//         llama_backend_init();
//         let sampler = llama_sampler_init_temp_ext(0.8, 0.01, 1.0);
//         let model_path = CString::new("models/llama-2-7b-chat.Q4_0.gguf").unwrap();
//         let model_params = llama_model_default_params();
//         let model = llama_load_model_from_file(model_path.as_ptr(), model_params);
//         let ctx_params = llama_context_default_params();
//         let ctx = llama_new_context_with_model(model, ctx_params);
//         let vocab = llama_model_get_vocab(model);
//
//         // Example CSV file and query
//         let csv_path = "data/sales.csv";
//         let query = "Return only the top 2 sales.";
//
//         run_csv_query(ctx, vocab, sampler, csv_path, query, "output.txt").unwrap();
//
//         llama_sampler_free(sampler);
//         llama_free(ctx);
//         llama_free_model(model);
//     }
// }
// Chat REPL: cargo run

// From file: cargo run -- --file prompts.txt        — reads prompts.txt and generates output for the whole text.

// From prompt: cargo run -- --prompt "Explain Rust ownership"        — generates output for that prompt once.

// From csv file: cargo run  -- --csv ./data/sales.csv $'Given the following CSV data:\n{csv}\n\nQuestion: List all unique customers\n\nReturn ONLY the list of customer names, one per line, with no additional text or explanations.' ./output.txt

// - Loading a local LLaMA model from models/llama-2-7b-chat.Q4_0.gguf using the C API (llama.cpp library) via Rust's extern "C" FFI.
//
// - Initializing a sampler (you’re switching between temperature sampling and greedy sampling).
//
// - Taking user input from the terminal (> prompt).
//
// - Tokenizing the input into model tokens (llama_tokenize).
//
// - Feeding tokens to the model (llama_decode).
//
// - Generating a sequence of tokens in a loop (max 50 tokens here).
//
// - Converting tokens to text (llama_vocab_get_text) and printing them to stdout.
//
// - Repeating until I type "exit".
//
// So it’s a minimal interactive text generation REPL for LLaMA models, similar to the official llama-cli but written entirely in Rust and FFI bindings.

// +--------------------+         FFI bindings         +-------------------+
// |                    | <---------------------------->|                   |
// |      Rust Code   | calls C functions via extern  |    llama.cpp C    |
// |  (main.rs & async) |       "C" ABI (FFI)           |    model & API    |
// |                    |                              |                   |
// +--------------------+                              +-------------------+
// |                                                      ^
// |                                                      |
// | User types prompt                                    |
// |                                                      |
// v                                                      |
// Interactive CLI REPL (loop)                                   |
// |                                                      |
// |    Tokens and logits sent back and forth             |
// +------------------------------------------------------+
