// llama.rs
use std::ffi::CString;
use std::{io, ptr};
use std::fs::File;
use std::io::Write;
use std::os::raw::c_char;
use rusty_llama::ffi::*;
use clap::Subcommand;
use std::ffi::CStr;
use std::slice;
pub struct Model {
    pub ptr: *mut LlamaModel,
}

impl Model {
    pub fn load(path: &str, params: LlamaModelParams) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path string")?;
        let model_ptr = unsafe { llama_load_model_from_file(c_path.as_ptr(), params) };

        if model_ptr.is_null() {
            Err("Failed to load model".into())
        } else {
            Ok(Model { ptr: model_ptr })
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { llama_free_model(self.ptr) };
    }
}

pub struct Context {
    pub ptr: *mut LlamaContext,
}

impl Context {
    pub fn new(model: &Model, params: LlamaContextParams) -> Result<Self, String> {
        let ctx_ptr = unsafe { llama_new_context_with_model(model.ptr, params) };
        if ctx_ptr.is_null() {
            Err("Failed to create context".into())
        } else {
            Ok(Self { ptr: ctx_ptr })
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { llama_free(self.ptr) };
    }
}

use clap::{Parser};

#[derive(Parser)]
#[command(author, version, about)]
pub struct Cli {
    #[arg(short, long, default_value = "models/llama-2-7b-chat.Q4_0.gguf")]
    pub model: String,

    /// Temperature for sampling (e.g. 0.8)
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Top-k sampling (e.g. 50)
    #[arg(long, default_value_t = 0)]
    pub top_k: i32,

    /// Top-p sampling (nucleus) (e.g. 0.9)
    #[arg(long, default_value_t = 0.0)]
    pub top_p: f32,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Interactive chat mode
    Chat,
    // Chat {
    /// If set, the model will answer briefly with no explanations
    // #[arg(short, long)]
    // brief: bool,
    // },
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
pub fn setup_sampler(cli: &Cli) -> *mut LlamaSampler {
    // Initialize default sampler chain params
    let params = unsafe { llama_sampler_chain_default_params() };
    let chain = unsafe { llama_sampler_chain_init(params) };

    // Add top_k sampler if top_k > 0
    if cli.top_k > 0 {
        let top_k_sampler = unsafe {  llama_sampler_init_top_k(cli.top_k as i32) };
        unsafe { llama_sampler_chain_add(chain, top_k_sampler)} ;
    }

    // Add top_p sampler if top_p is between 0 and 1
    if cli.top_p > 0.0 && cli.top_p < 1.0 {
        let top_p_sampler = unsafe {  llama_sampler_init_top_p(cli.top_p, 1) };
            unsafe {  llama_sampler_chain_add(chain, top_p_sampler) };
    }

    // Add temperature sampler if temperature > 0
    if cli.temperature > 0.0 {
        let temp_sampler = unsafe {  llama_sampler_init_temp(cli.temperature) };
            unsafe {  llama_sampler_chain_add(chain, temp_sampler) };
    }

    use rand::Rng;

    let seed: u32 = rand::thread_rng().gen();
    let dist_sampler = unsafe {  llama_sampler_init_dist(seed) };

    // Always add a final sampler to pick actual tokens (greedy here)
    let greedy_sampler = unsafe {  llama_sampler_init_greedy()};
    unsafe {  llama_sampler_chain_add(chain, greedy_sampler) };

    chain
}
/// Initialize library/backend
pub fn backend_init() {
    unsafe { llama_backend_init() }
}

/// Return default model params (by-value). This is safe.
pub fn model_default_params() -> LlamaModelParams {
    // Safe to call, returns by value
    unsafe { llama_model_default_params() }
}

/// Return default context params
pub fn context_default_params() -> LlamaContextParams {
    unsafe { llama_context_default_params() }
}


pub fn sampler_free(sampler: *mut LlamaSampler) {
    if !sampler.is_null() {
        unsafe { llama_sampler_free(sampler) }
    }
}

pub fn free_context(ctx: *mut LlamaContext) {
    if !ctx.is_null() {
        unsafe { llama_free(ctx) }
    }
}

pub fn free_model(model: *mut LlamaModel) {
    if !model.is_null() {
        unsafe { llama_free_model(model) }
    }
}

pub fn load_model_from_file(path: &CStr, params: LlamaModelParams) -> Result<*mut LlamaModel, String> {
    let ptr = unsafe { llama_load_model_from_file(path.as_ptr(), params) };
    if ptr.is_null() {
        Err("Failed to load model: null pointer".to_string())
    } else {
        Ok(ptr)
    }
}

pub fn new_context_with_model(model: *mut LlamaModel, params: LlamaContextParams) -> Result<*mut LlamaContext, String> {
    let ctx = unsafe { llama_new_context_with_model(model, params) };
    if ctx.is_null() {
        Err("Failed to create context: null pointer".to_string())
    } else {
        Ok(ctx)
    }
}
// Similarly add safe wrappers for other pointer-returning or unsafe functions.

// For example:

pub fn model_get_vocab(model: *const LlamaModel) -> Result<*const LlamaVocab, String> {
    let vocab = unsafe { llama_model_get_vocab(model) };
    if vocab.is_null() {
        Err("Failed to get vocab: null pointer".to_string())
    } else {
        Ok(vocab)
    }
}
pub fn vocab_get_text(vocab: *const LlamaVocab, token: i32) -> Result<&'static CStr, String> {
    let ptr = unsafe { llama_vocab_get_text(vocab, token) };
    if ptr.is_null() {
        Err("Failed to get token text: null pointer".to_string())
    } else {
        // SAFETY: we trust the C string returned is valid and null-terminated
        let cstr = unsafe { CStr::from_ptr(ptr) };
        Ok(cstr)
    }
}

pub fn run_csv_query(
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

fn safe_get_logits<'a>(
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
) -> Result<&'a [f32], String> {
    if ctx.is_null() {
        return Err("Context pointer is null".into());
    }
    if vocab.is_null() {
        return Err("Vocab pointer is null".into());
    }

    let ptr = unsafe { llama_get_logits(ctx) };
    if ptr.is_null() {
        return Err("Logits pointer is null".into());
    }

    let vocab_size = unsafe { llama_vocab_n_tokens(vocab) };
    Ok(unsafe { std::slice::from_raw_parts(ptr, vocab_size as usize) })
}
pub fn get_logits(ctx: *mut LlamaContext, vocab_size: usize) -> Result<Vec<f32>, String> {
    if ctx.is_null() {
        return Err("Context is null".into());
    }

    let ptr = unsafe { llama_get_logits(ctx) };
    if ptr.is_null() {
        return Err("Failed to get logits".into());
    }

    // Safety: llama_get_logits returns a pointer to logits array of length vocab_size.
    let slice = unsafe { slice::from_raw_parts(ptr, vocab_size) };

    // Copy into a Vec to avoid lifetime/dangling pointer issues
    Ok(slice.to_vec())
}
pub fn generate_text(
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    sampler: *mut LlamaSampler,
    prompt: &str,
) -> String {

    unsafe {
        // Reset the model's memory to clear past context
        let mem = llama_get_memory(ctx);
        assert!(!mem.is_null(), "Memory pointer is null");
        llama_memory_clear(mem, true);
    }

    let full_prompt = with_instruction(&prompt);
    // Tokenize the prompt string into model tokens
    // let prompt_c = CString::new(prompt).unwrap();
    let prompt_c = CString::new(full_prompt.clone()).unwrap();
    let mut tokens = [0i32; 512];

    let n_tokens = unsafe {
        llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            full_prompt.len() as i32,
            // prompt.len() as i32,
            // prompt.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,  // add_special tokens
            false, // parse_special tokens
        )
    };
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
    unsafe {
        let ret = llama_decode(ctx, batch);
        assert_eq!(ret, 0, "Decode failed on prompt tokens");
    }


    // Prepare to generate tokens for completion
    let max_tokens = 250;

    let mut cur_len = n_tokens as usize;
    print!("→ ");
    io::stdout().flush().unwrap();

    let mut n_past = n_tokens;
    let mut generated_tokens = Vec::new();
    let mut output = String::new();
    let mut newline_count = 0;
    let stop_tokens = vec![2 /* EOS */, 26077 /* " unwanted token IDs here */];


    for _ in 0..max_tokens {
        let next_token = generated_tokens.last().unwrap_or(&tokens[n_past as usize - 1]);
        // implement breaking after 3 new lines
        // if stop_tokens.contains(&next_token) {
        //     break;
        // }
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
        unsafe {
            let ret = unsafe {  llama_decode(ctx, batch) };
            assert_eq!(ret, 0, "Decode failed");

            let logits_ptr = unsafe {  llama_get_logits(ctx) };
            let logits_ptr = unsafe {  llama_get_logits(ctx) };
            assert!(!logits_ptr.is_null(), "Logits pointer is null");

            let vocab_size = unsafe {  llama_vocab_n_tokens(vocab) };

            // Wrap logits into token data for sampler
            let mut token_data: Vec<LlamaTokenData> = (0..vocab_size).map(|i| {
                LlamaTokenData {
                    id: i as i32,
                    logit: *logits_ptr.add(i as usize),
                    p: 0.0,
                }
            }).collect();

            let mut token_data_array = unsafe {
                LlamaTokenDataArray {
                    data: token_data.as_mut_ptr(),
                    size: token_data.len(),
                    sorted: false,
                }
            };

            // Apply sampling to modify logits (e.g., greedy, temperature, top-p)
            unsafe {  llama_sampler_apply(sampler, &mut token_data_array) };

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

            let token_text_ptr = unsafe {  llama_vocab_get_text(vocab, next_token) };

            let mut token_text = std::ffi::CStr::from_ptr(token_text_ptr)
                .to_string_lossy()
                .into_owned();

            // Replace visible hex newlines with real ones and clean up whitespace token
            token_text = token_text.replace("<0x0A>", "\n");
            token_text = token_text.replace('▁', " ");
            print!("{}", token_text);
            io::stdout().flush().unwrap();
            output.push_str(&token_text);

            // Stop generation if EOS or BOS tokens appear -- doesn't work now as often have </s> token
            // if next_token == 2 || next_token == 0 {
            //     break;
            // }

            n_past += 1;
            generated_tokens.push(next_token);
        }
    }
    let output = output.trim();

    let output = output.trim_end_matches(|c: char| {
        c.is_whitespace() || c == '.' || c == ',' || c == '\n' || c == '\r'
    });
    println!();

    output.to_string()
}

pub fn with_instruction(prompt: &str) -> String {
    format!(
        "You are a precise assistant. Answer the Question:\n\n{}",
        prompt
    )
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
    };

    Ok(full_csv)
}