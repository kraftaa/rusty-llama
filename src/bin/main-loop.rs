use std::io;
use std::io::Write;
use std::ffi::{CString, CStr};
use std::os::raw::{c_int, c_float, c_char, c_void};

type LlamaToken = i32; // confirm actual type in your headers
type LlamaPos = i32;   // adjust if different
type LlamaSeqId = i32; // adjust if different
use std::ptr;

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

// #[repr(C)]
// pub struct llama_memory_t {
//     _private: [u8; 0],
// }
#[repr(C)]
struct LlamaMemory {
    _private: [u8; 0], // opaque struct
}
// pub type llama_memory_t = llama_memory;

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

fn main() {
    unsafe {
        // Initialize llama backend (required before any model loading or inference)
        llama_backend_init();
        // let sampler = llama_sampler_init_temp(0.8);
        let sampler = llama_sampler_init_temp_ext(0.8, 0.01, 1.0);
        println!("sampler {:?}", &sampler.as_ref().unwrap());
        assert!(!sampler.is_null(), "Failed to init sampler");
        // Load model from file
        // let model_path = CString::new("models/ggml-model-q4_0.gguf").unwrap();
        let model_path = CString::new("models/llama-2-7b-chat.Q4_0.gguf").unwrap();
        let model_params = llama_model_default_params();
        let model = llama_load_model_from_file(model_path.as_ptr(), model_params);
        assert!(!model.is_null(), "Failed to load model");

        // Create inference context
        let ctx_params = llama_context_default_params();
        let ctx = llama_new_context_with_model(model, ctx_params);
        assert!(!ctx.is_null(), "Failed to create context");

        // Get vocab from model (older API)
        let vocab = llama_model_get_vocab(model);
        assert!(!vocab.is_null(), "Failed to get vocab");

        // Initialize greedy sampler (picks highest probability token each time)
        let sampler = llama_sampler_init_greedy();
        // let sampler = llama_sampler_init_temp_ext(0.8, 0.01, 1.0);
        // let sampler = llama_sampler_init_temp(0.8);
        assert!(!sampler.is_null(), "Failed to init sampler");

        let mut n_past = 0;

        loop {
            // Ask user for input
            print!("\n> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input == "exit" {
                break;
            }
            // Reset KV cache for sequence 0 to avoid position mismatch
            // llama_get_memory(ctx, 0);
            // let mem = llama_get_memory(ctx);
            // llama_memory_clear(mem, true);
            // let mut mem = llama_get_memory(ctx);  // get the struct by value
            // llama_memory_clear(&mut mem as *mut llama_memory_t, true);
            let mem = llama_get_memory(ctx);
            assert!(!mem.is_null(), "Memory pointer is null");
            llama_memory_clear(mem, true);
            n_past = 0;
            // llama_get_memory(ctx);
            // n_past = 0; // Also reset n_past to start from position 0

            // Convert prompt to C string
            let prompt_c = CString::new(input).unwrap();
            let mut tokens = [0i32; 512];

            // Tokenize input into llama tokens
            let n_tokens = llama_tokenize(
                vocab,
                prompt_c.as_ptr(),
                input.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,  // add_special
                false, // parse_special
            );
            assert!(n_tokens > 0, "Tokenization failed");

            // Feed the entire prompt tokens at once with consecutive positions
            {
                // let pos: Vec<i32> = (n_past..n_past + n_tokens).collect();
                let pos: Vec<i32> = (0..n_tokens).collect();
                let mut batch = LlamaBatch {
                    n_tokens,
                    token: tokens.as_mut_ptr(),
                    embd: ptr::null_mut(),
                    pos: pos.as_ptr() as *mut i32,
                    n_seq_id: ptr::null_mut(),
                    seq_id: ptr::null_mut(),
                    logits: ptr::null_mut(), // no logits requested for prompt feed
                };
                let ret = llama_decode(ctx, batch);
                assert_eq!(ret, 0, "Decode failed on prompt tokens");
                n_past += n_tokens;
            }
            // n_past += n_tokens;

            let max_tokens = 50;
            let mut cur_len = n_tokens as usize;
            let mut generated_tokens = Vec::new();
            // let mut generated_tokens = vec![tokens[n_past as usize - 1]];
            print!("→ ");
            io::stdout().flush().unwrap();


            for i in 0..max_tokens {
                // Decode only the last token to get logits for next prediction
                // let token_slice = &mut tokens[cur_len - 1..cur_len];
                // let mut pos = [(cur_len - 1) as i32];
                let next_token = generated_tokens.last().unwrap_or(&tokens[n_past as usize - 1]);
                // let last_token = if i == 0 {
                //     // no tokens generated yet, you can pick a dummy token or last prompt token
                //     // tokens[n_past as usize - 1]
                //     // Use a safe default like the last prompt token or a special token
                //     tokens.get(n_past as usize - 1).copied().unwrap_or(0) //.expect("Prompt tokens empty")
                // } else {
                //     // *generated_tokens.last().unwrap()
                //     generated_tokens.last().copied().unwrap_or(0) // .expect("No tokens generated yet")
                // };
                let mut token_slice = [*next_token];
                // let mut token_slice = &mut tokens[cur_len - 1..cur_len];
                let mut pos = [n_past];
                // let mut logits_required = [true as i8];
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

                // Get raw logits from model
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

                // Apply greedy sampling (select most probable next token)
                llama_sampler_apply(sampler, &mut token_data_array);

                // let next_token = llama_sampler_sample(sampler, ctx, 0);
                let next_token = unsafe {
                    llama_sampler_sample(sampler, ctx, 0)
                };

                llama_sampler_accept(sampler, next_token);

                // Append new token
                if cur_len < tokens.len() {
                    tokens[cur_len] = next_token;
                    cur_len += 1;
                    // n_past += 1;
                } else {
                    println!("\n[Token buffer full]");
                    break;
                }

                // Convert token to readable string and print
                let token_text_ptr = llama_vocab_get_text(vocab, next_token);

                let mut token_text = std::ffi::CStr::from_ptr(token_text_ptr)
                    .to_string_lossy()
                    .into_owned();

                // Replace visible hex newlines with real ones
                token_text = token_text.replace("<0x0A>", "\n");
                token_text = token_text.replace('▁', " ");
                print!("{}", token_text);

                io::stdout().flush().unwrap();

                // Stop if end-of-sequence token is generated (EOS or BOS)
                if next_token == 2 || next_token == 0 {
                    break;
                }
                n_past += 1;
            }

            println!();
        }

        // Clean up
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_free_model(model);
    }
}

// working
// fn main() {
//     unsafe {
//         llama_backend_init();
//
//         // let model = llama_load_model_from_file(...);
//         let model_path = CString::new("models/ggml-model-q4_0.gguf").unwrap();
//         let model_params = llama_model_default_params();
//         let model = llama_load_model_from_file(model_path.as_ptr(), model_params);
//
//         assert!(!model.is_null(), "Failed to load model");
//         // Create inference context
//         let ctx_params = llama_context_default_params();
//         let ctx = llama_new_context_with_model(model, ctx_params);
//         assert!(!ctx.is_null(), "Failed to create context");
//
//         // Initialize sampler AFTER model and context are ready
//         let sampler = llama_sampler_init_temp(0.8);
//         assert!(!sampler.is_null());
//
//         // Use sampler for inference...
//
//         llama_sampler_free(sampler);
//         llama_free(ctx);
//         llama_free_model(model);
//     }
// }
// working with greedy sampler still breaking with temp
// fn main() {
//     unsafe {
//         llama_backend_init();
//
//         let model_path = CString::new("models/ggml-model-q4_0.gguf").unwrap();
//         let model_params = llama_model_default_params();
//         let model = llama_load_model_from_file(model_path.as_ptr(), model_params);
//         assert!(!model.is_null(), "Failed to load model");
//
//         let ctx_params = llama_context_default_params();
//         let ctx = llama_new_context_with_model(model, ctx_params);
//         assert!(!ctx.is_null(), "Failed to create context");
//
//         let vocab = llama_model_get_vocab(model);
//         assert!(!vocab.is_null(), "Failed to get vocab");
//         let sampler = llama_sampler_init_greedy();
//         // let sampler = llama_sampler_init_temp(0.8);
//         assert!(!sampler.is_null(), "Failed to init sampler");
//
//         let mut n_past = 0;
//
//         loop {
//             print!("> ");
//             io::stdout().flush().unwrap();
//
//             let mut input = String::new();
//             io::stdin().read_line(&mut input).unwrap();
//             let input = input.trim();
//
//             if input == "exit" {
//                 break;
//             }
//
//             // Tokenize input
//             let prompt_c = CString::new(input).unwrap();
//             let mut tokens = [0i32; 512];
//             let n_tokens = llama_tokenize(
//                 vocab,
//                 prompt_c.as_ptr(),
//                 input.len() as i32,
//                 tokens.as_mut_ptr(),
//                 tokens.len() as i32,
//                 true,
//                 false,
//             );
//             assert!(n_tokens > 0, "Tokenization failed");
//
//             // Feed prompt tokens
//             println!("Feeding prompt tokens, n_tokens = {}, n_past = {}", n_tokens, n_past);
//
//             let pos: Vec<i32> = (n_past..n_past + n_tokens).collect();
//             println!("Positions (first 5): {:?}", &pos[0..std::cmp::min(5, pos.len())]);
//
//             let mut batch = LlamaBatch {
//                 n_tokens,
//                 token: tokens.as_mut_ptr(),
//                 embd: ptr::null_mut(),
//                 pos: pos.as_ptr() as *mut i32,
//                 n_seq_id: ptr::null_mut(),
//                 seq_id: ptr::null_mut(),
//                 logits: ptr::null_mut(),
//             };
//
//             let ret = llama_decode(ctx, batch);
//             assert_eq!(ret, 0, "Decode failed on prompt tokens");
//             n_past += n_tokens;
//
//             // Generate tokens
//             let max_tokens = 50;
//             for i in 0..max_tokens {
//                 let last_token = tokens[n_past as usize - 1];
//
//                 let mut token_slice = [last_token];
//                 let mut pos = [n_past];
//                 let mut logits_required = [true as i8];
//
//                 let mut batch = LlamaBatch {
//                     n_tokens: 1,
//                     token: token_slice.as_mut_ptr(),
//                     embd: ptr::null_mut(),
//                     pos: pos.as_ptr() as *mut i32,
//                     n_seq_id: ptr::null_mut(),
//                     seq_id: ptr::null_mut(),
//                     logits: logits_required.as_mut_ptr(),
//                 };
//
//                 let ret = llama_decode(ctx, batch);
//                 assert_eq!(ret, 0, "Decode failed");
//
//                 let logits_ptr = llama_get_logits(ctx);
//                 assert!(!logits_ptr.is_null(), "Logits pointer is null");
//
//                 // let vocab_size = llama_vocab_n_tokens(vocab);
//                 let vocab_size = llama_vocab_n_tokens(vocab) as usize;
//
//
//                 let mut token_data: Vec<LlamaTokenData> = (0..vocab_size)
//                     .map(|i| LlamaTokenData {
//                         id: i as i32,
//                         logit: *logits_ptr.add(i as usize),
//                         p: 0.0,
//                     })
//                     .collect();
//
//                 println!("vocab_size = {}", vocab_size);
//                 println!("First 5 logits:");
//                 for i in 0..std::cmp::min(5, vocab_size) {
//                     println!("  token_id {} logit = {}", i, token_data[i].logit);
//                 }
//
//                 let mut token_data_array = LlamaTokenDataArray {
//                     data: token_data.as_mut_ptr(),
//                     size: token_data.len(),
//                     sorted: false,
//                 };
//
//                 println!("token_data_array.size = {}", token_data_array.size);
//
//                 println!("Applying sampler to token_data_array with size = {}", token_data_array.size);
//
//                 llama_sampler_apply(sampler, &mut token_data_array);
//                 println!("Sampling next token...");
//                 let next_token = llama_sampler_sample(sampler, ctx, 0);
//                 println!("Sampled next_token = {}", next_token);
//                 llama_sampler_accept(sampler, next_token);
//
//                 if next_token == 2 || next_token == 0 {
//                     break;
//                 }
//
//                 // Append token to tokens array and increment
//                 tokens[n_past as usize] = next_token;
//                 n_past += 1;
//
//                 // Print generated token text
//                 let token_text_ptr = llama_vocab_get_text(vocab, next_token);
//                 let token_text = std::ffi::CStr::from_ptr(token_text_ptr).to_string_lossy();
//                 print!("{}", token_text);
//                 io::stdout().flush().unwrap();
//             }
//             println!();
//         }
//
//         llama_sampler_free(sampler);
//         llama_free(ctx);
//         llama_free_model(model);
//     }
// }
// Applying sampler to token_data_array with size = 32000
// Sampling next token...
// Sampled next_token = 7251
// ▁hivocab_size = 32000
// First 5 logits:
// token_id 0 logit = -3.4185784
// token_id 1 logit = -2.9577909
// token_id 2 logit = 13.054774
// token_id 3 logit = 6.5687547
// token_id 4 logit = 4.159735
// token_data_array.size = 32000
// Applying sampler to token_data_array with size = 32000
// Sampling next token...
// Sampled next_token = 7251
// ▁hi

