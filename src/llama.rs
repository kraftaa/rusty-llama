use std::ffi::{CString, CStr};
use std::{io, ptr};
use std::fs::File;
use std::io::Write;
use std::slice;

use crate::ffi::*;
use clap::{Parser, Subcommand};

use std::ptr::NonNull;

// pub struct Model {
//     ptr: NonNull<LlamaModel>,
//     _path_cstring: CString, // keep CString alive so pointer remains valid
// }
// #[derive(Debug)]
// impl Model {
//     pub fn load(path: &str, params: LlamaModelParams) -> Result<Self, String> {
//         let c_path = CString::new(path).map_err(|_| "Invalid path string")?;
//         // Call FFI function inside unsafe block
//         let model_ptr = unsafe { llama_load_model_from_file(c_path.as_ptr(), params) };
//         let ptr = NonNull::new(model_ptr).ok_or_else(|| "Failed to load model: null pointer".to_string())?;
//         Ok(Model { ptr, _path_cstring: c_path })
//     }
//
//     pub fn get_vocab(&self) -> Result<NonNull<LlamaVocab>, String> {
//         let vocab_ptr = unsafe { llama_model_get_vocab(self.ptr.as_ptr()) };
//         NonNull::new(vocab_ptr).ok_or_else(|| "Failed to get vocab: null pointer".to_string())
//     }
//
//     pub fn raw_ptr(&self) -> *mut LlamaModel {
//         self.ptr.as_ptr()
//     }
// }

// impl Drop for Model {
//     fn drop(&mut self) {
//         unsafe {
//             if !self.ptr.is_null() {
//                 llama_free_model(self.ptr);
//             }
//         }
//     }
// }
// #[derive(Debug)]
// pub struct Context {
//     ptr: NonNull<LlamaContext>,
// }

// impl Context {
//     pub fn new(model: &Model, params: LlamaContextParams) -> Result<Self, String> {
//         let ctx_ptr = unsafe { llama_new_context_with_model(model.raw_ptr(), params) };
//         let ptr = NonNull::new(ctx_ptr).ok_or_else(|| "Failed to create context: null pointer".to_string())?;
//         Ok(Context { ptr })
//     }
//
//     pub fn raw_ptr(&self) -> *mut LlamaContext {
//         self.ptr.as_ptr()
//     }
// }

// impl Drop for Context {
//     fn drop(&mut self) {
//         unsafe { llama_free(self.ptr.as_ptr()) }
//     }
// }


#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Cli {
    #[arg(short, long, default_value = "models/llama-2-7b-chat.Q4_0.gguf")]
    pub model: String,

    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    #[arg(long, default_value_t = 0)]
    pub top_k: i32,

    #[arg(long, default_value_t = 0.0)]
    pub top_p: f32,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Chat,
    File { filename: String },
    Prompt { text: Vec<String> },
    Csv {
        csv_path: String,
        output_path: String,
        query: Vec<String>,
    },
}

pub fn setup_sampler(cli: &Cli) -> Result<*mut LlamaSampler, String> {
    let params = unsafe { llama_sampler_chain_default_params() };
    let chain = unsafe { llama_sampler_chain_init(params) };
    if chain.is_null() {
        return Err("Failed to initialize sampler chain".into());
    }

    if cli.top_k > 0 {
        let top_k_sampler = unsafe { llama_sampler_init_top_k(cli.top_k) };
        if top_k_sampler.is_null() {
            return Err("Failed to initialize top_k sampler".into());
        }
        unsafe { llama_sampler_chain_add(chain, top_k_sampler) };
    }

    if cli.top_p > 0.0 && cli.top_p < 1.0 {
        let top_p_sampler = unsafe { llama_sampler_init_top_p(cli.top_p, 1) };
        if top_p_sampler.is_null() {
            return Err("Failed to initialize top_p sampler".into());
        }
        unsafe { llama_sampler_chain_add(chain, top_p_sampler) };
    }

    if cli.temperature > 0.0 {
        let temp_sampler = unsafe { llama_sampler_init_temp(cli.temperature) };
        if temp_sampler.is_null() {
            return Err("Failed to initialize temperature sampler".into());
        }
        unsafe { llama_sampler_chain_add(chain, temp_sampler) };
    }

    use rand::Rng;
    let seed: u32 = rand::thread_rng().gen();
    let dist_sampler = unsafe { llama_sampler_init_dist(seed) };
    if dist_sampler.is_null() {
        return Err("Failed to initialize dist sampler".into());
    }
    unsafe { llama_sampler_chain_add(chain, dist_sampler) };

    let greedy_sampler = unsafe { llama_sampler_init_greedy() };
    if greedy_sampler.is_null() {
        return Err("Failed to initialize greedy sampler".into());
    }
    unsafe { llama_sampler_chain_add(chain, greedy_sampler) };

    Ok(chain)
}

pub fn sampler_free(sampler: *mut LlamaSampler) {
    if !sampler.is_null() {
        unsafe { llama_sampler_free(sampler) }
    }
}

pub fn model_default_params() -> LlamaModelParams {
    unsafe { llama_model_default_params() }
}

pub fn context_default_params() -> LlamaContextParams {
    unsafe { llama_context_default_params() }
}

pub fn generate_text(
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    sampler: *mut LlamaSampler,
    prompt: &str,
) -> Result<String, String> {
    if ctx.is_null() || vocab.is_null() || sampler.is_null() {
        return Err("Null pointer passed to generate_text".into());
    }

    unsafe {
        let mem = llama_get_memory(ctx);
        if mem.is_null() {
            return Err("Null memory pointer".into());
        }
        llama_memory_clear(mem, true);
    }

    let full_prompt = with_instruction(prompt);
    let prompt_c = CString::new(full_prompt.clone()).map_err(|_| "Failed to convert prompt")?;

    let mut tokens = [0i32; 512];
    let n_tokens = unsafe {
        llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            full_prompt.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,
            false,
        )
    };
    if n_tokens <= 0 {
        return Err("Tokenization failed".into());
    }

    println!("Token count: {}", n_tokens);

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
        if ret != 0 {
            return Err("Decode failed on prompt tokens".into());
        }
    }

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

        if *next_token == 13 {
            newline_count += 1;
            if newline_count >= 3 {
                break;
            }
        } else {
            newline_count = 0;
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
            let ret = llama_decode(ctx, batch);
            if ret != 0 {
                return Err("Decode failed".into());
            }

            let logits_ptr = llama_get_logits(ctx);
            if logits_ptr.is_null() {
                return Err("Logits pointer is null".into());
            }

            let vocab_size = llama_vocab_n_tokens(vocab) as usize;

            let mut token_data: Vec<LlamaTokenData> = (0..vocab_size)
                .map(|i| LlamaTokenData {
                    id: i as i32,
                    logit: *logits_ptr.add(i),
                    p: 0.0,
                })
                .collect();

            let mut token_data_array = LlamaTokenDataArray {
                data: token_data.as_mut_ptr(),
                size: token_data.len(),
                sorted: false,
            };

            llama_sampler_apply(sampler, &mut token_data_array);

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
            if token_text_ptr.is_null() {
                return Err("Failed to get token text".into());
            }

            let mut token_text = CStr::from_ptr(token_text_ptr)
                .to_string_lossy()
                .into_owned();

            token_text = token_text.replace("<0x0A>", "\n");
            token_text = token_text.replace('▁', " ");

            print!("{}", token_text);
            io::stdout().flush().unwrap();
            output.push_str(&token_text);

            n_past += 1;
            generated_tokens.push(next_token);
        }
    }

    let output = output.trim();
    let output = output.trim_end_matches(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == '\n' || c == '\r');

    println!();

    Ok(output.to_string())
}

pub fn with_instruction(prompt: &str) -> String {
    format!("You are a precise assistant. Answer the Question:\n\n{}", prompt)
}

fn read_csv_with_header(csv_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(csv_path)?;
    let first_line = content.lines().next().unwrap_or("");
    let has_header = first_line.chars().any(|c| c.is_alphabetic());

    if has_header {
        Ok(content)
    } else {
        println!("No headers in the file");
        Ok("No headers in the file".to_string())
    }
}

pub fn run_csv_query(
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    sampler: *mut LlamaSampler,
    csv_path: &str,
    prompt_template: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let csv_text = read_csv_with_header(csv_path)?;
    println!("CSV content:\n{}", csv_text);

    let prompt_template = prompt_template.replace("\\n", "\n");
    let prompt = prompt_template.replace("{csv}", &csv_text);
    println!("Final prompt sent to model:\n{}", prompt);

    let generated = generate_text(ctx, vocab, sampler, &prompt)
        .map_err(|e| format!("Generation error: {}", e))?;

    let mut file = File::create(output_path)?;
    file.write_all(generated.as_bytes())?;

    println!("Generated output saved to {}", output_path);

    Ok(())
}
#[derive(Debug)]
pub struct Sampler {
    ptr: *mut LlamaSampler,
}

impl Sampler {
    pub fn new(cli: &Cli) -> Result<Self, String> {
        // Initialize sampler chain with default params
        let chain = unsafe { llama_sampler_chain_init(unsafe { llama_sampler_chain_default_params() }) };

        if chain.is_null() {
            return Err("Failed to init sampler chain".into());
        }

        // Add top-k sampler if enabled
        if cli.top_k > 0 {
            let top_k_sampler = unsafe { llama_sampler_init_top_k(cli.top_k as i32) };
            unsafe { llama_sampler_chain_add(chain, top_k_sampler) };
        }

        // Add top-p sampler if enabled and valid
        if cli.top_p > 0.0 && cli.top_p < 1.0 {
            let top_p_sampler = unsafe { llama_sampler_init_top_p(cli.top_p, 1) };
            unsafe { llama_sampler_chain_add(chain, top_p_sampler) };
        }

        // Add temperature sampler if enabled
        if cli.temperature > 0.0 {
            let temp_sampler = unsafe { llama_sampler_init_temp(cli.temperature) };
            unsafe { llama_sampler_chain_add(chain, temp_sampler) };
        }

        // Add distribution sampler seeded randomly
        use rand::Rng;
        let seed: u32 = rand::thread_rng().gen();
        let dist_sampler = unsafe { llama_sampler_init_dist(seed) };
        unsafe { llama_sampler_chain_add(chain, dist_sampler) };

        // Add greedy sampler last
        let greedy_sampler = unsafe { llama_sampler_init_greedy() };
        unsafe { llama_sampler_chain_add(chain, greedy_sampler) };

        Ok(Sampler { ptr: chain })
    }
    pub fn ptr(&self) -> *mut LlamaSampler {
        self.ptr
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { llama_sampler_free(self.ptr) };
        }
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

pub fn backend_init() {
    unsafe { llama_backend_init() }
}

pub fn print_model_params(params: &LlamaModelParams) {
    println!("LlamaModelParams {{");
    println!("  n_gpu_layers: {}", params.n_gpu_layers);
    println!("  main_gpu: {}", params.main_gpu);
    println!("  tensor_split: {:?}", params.tensor_split);
    println!("  tensor_split_size: {}", params.tensor_split_size);
    println!("  vocab_only: {}", params.vocab_only);
    println!("  use_mmap: {}", params.use_mmap);
    println!("  use_mlock: {}", params.use_mlock);
    println!("  check_tensors: {}", params.check_tensors);
    println!("  progress_callback: {:?}", params.progress_callback);
    println!("  progress_callback_user_data: {:?}", params.progress_callback_user_data);
    println!("}}");
}

pub fn print_context_params(params: &LlamaContextParams) {
    println!("LlamaContextParams {{");
    println!("  seed: {}", params.seed);
    println!("  n_ctx: {}", params.n_ctx);
    println!("  n_batch: {}", params.n_batch);
    println!("  n_threads: {}", params.n_threads);
    println!("  n_threads_batch: {}", params.n_threads_batch);
    println!("  rope_freq_base: {}", params.rope_freq_base);
    println!("  rope_freq_scale: {}", params.rope_freq_scale);
    println!("  yarn_ext_factor: {}", params.yarn_ext_factor);
    println!("  yarn_attn_factor: {}", params.yarn_attn_factor);
    println!("  yarn_beta_fast: {}", params.yarn_beta_fast);
    println!("  yarn_beta_slow: {}", params.yarn_beta_slow);
    println!("  yarn_orig_ctx: {}", params.yarn_orig_ctx);
    println!("  mul_mat_q: {}", params.mul_mat_q);
    println!("  f16_kv: {}", params.f16_kv);
    println!("  logits_all: {}", params.logits_all);
    println!("  embedding: {}", params.embedding);
    println!("  offload_kqv: {}", params.offload_kqv);
    println!("  offload_kqv_linear: {}", params.offload_kqv_linear);
    println!("  flash_attn: {}", params.flash_attn);
    println!("  pooling_type: {}", params.pooling_type);
    println!("  rms_norm_eps: {}", params.rms_norm_eps);
    println!("}}");
}
pub fn model_get_vocab(model: *const LlamaModel) -> Result<*const LlamaVocab, String> {
    let vocab = unsafe { llama_model_get_vocab(model) };
    if vocab.is_null() {
        Err("Failed to get vocab: null pointer".to_string())
    } else {
        Ok(vocab)
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


// Safe Rust wrapper around LlamaModel raw pointer
pub struct Model {
    ptr: NonNull<LlamaModel>,
    _path_cstring: CString, // keep CString alive for path pointer validity
}

impl Model {
    pub fn load(path: &str, params: LlamaModelParams) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|_| "Invalid path string")?;
        let raw_ptr = unsafe { llama_load_model_from_file(c_path.as_ptr(), params) };

        let ptr = NonNull::new(raw_ptr).ok_or_else(|| "Failed to load model: null pointer".to_string())?;

        Ok(Model {
            ptr,
            _path_cstring: c_path,
        })
    }

    pub fn get_vocab(&self) -> Result<Vocab, String> {
        let vocab_ptr = unsafe { llama_model_get_vocab(self.ptr.as_ptr()) };
        let ptr = NonNull::new(vocab_ptr as *mut LlamaVocab)
            .ok_or_else(|| "Failed to get vocab: null pointer".to_string())?;
        Ok(Vocab { ptr })
    }

    pub fn ptr(&self) -> *mut LlamaModel {
        self.ptr.as_ptr()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            llama_free_model(self.ptr.as_ptr());
        }
    }
}

pub struct Context {
    ptr: NonNull<LlamaContext>,
}

impl Context {
    pub fn new(model: &Model, params: LlamaContextParams) -> Result<Self, String> {
        let ctx_ptr = unsafe { llama_new_context_with_model(model.ptr(), params) };
        let ptr = NonNull::new(ctx_ptr).ok_or_else(|| "Failed to create context: null pointer".to_string())?;
        Ok(Context { ptr })
    }

    pub fn ptr(&self) -> *mut LlamaContext {
        self.ptr.as_ptr()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.ptr.as_ptr());
        }
    }
}

pub struct Vocab {
    ptr: NonNull<LlamaVocab>,
}

impl Vocab {
    pub fn ptr(&self) -> *mut LlamaVocab {
        self.ptr.as_ptr()
    }
}
