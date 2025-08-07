
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
// pub struct LlamaSampler;  // Opaque pointer type
pub struct LlamaSampler {
    _private: [u8; 0], // opaque
}

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
    pub fn llama_sampler_sample(sampler: *mut LlamaSampler, ctx: *mut LlamaContext, idx: i32) -> i32;
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
    fn llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32) -> *mut LlamaSampler;
}
// unsafe extern "C" {
//     fn llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32) -> *mut LlamaSampler;
// }
fn main() {
    unsafe {
        llama_backend_init();

        // Load model
        let model_path = CString::new("models/ggml-model-q4_0.gguf").unwrap();
        let model_params = llama_model_default_params();
        let model = llama_load_model_from_file(model_path.as_ptr(), model_params);
        assert!(!model.is_null(), "Failed to load model");

        // Create context
        let ctx_params = llama_context_default_params();
        let ctx = llama_new_context_with_model(model, ctx_params);
        assert!(!ctx.is_null(), "Failed to create context");

        // Get vocab (older API)
        let vocab = llama_model_get_vocab(model);
        assert!(!vocab.is_null(), "Failed to get vocab");

        // --- Tokenize prompt ---
        let prompt_str = "Explain the following SQL query:\nSELECT * FROM users;";

        let prompt_c = CString::new(prompt_str).unwrap();
        let mut tokens = [0i32; 512];
        let n_tokens = llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            prompt_str.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,  // add_special
            false, // parse_special
        );
        assert!(n_tokens > 0, "Tokenization failed");

        // unsafe {
            // llama_sampler_init_temp_ext(ctx, 0.8);
            // Sampler
            let sampler = llama_sampler_init_greedy();
            // let sampler = llama_sampler_init_temp_ext(0.8, 0.01, 1.0);

            let temperature = 0.8;
            let delta = 0.01;
            let exponent = 1.0;

            // Call the C API to create a new sampler with temperature decay
            // let sampler = llama_sampler_init_temp_ext(temperature, delta, exponent);
            // assert!(!sampler.is_null(), "Failed to create sampler");
            assert!(!sampler.is_null(), "Failed to init sampler");

            // --- Generate ---
            let max_tokens = 20;
            let mut cur_len = n_tokens as usize;

            for _ in 0..max_tokens {
                // Setup batch with the *last* token only
                let token_slice = &mut tokens[cur_len - 1..cur_len];
                let mut pos = [(cur_len - 1) as i32];
                let mut logits_required = [true as i8];

                let mut batch = LlamaBatch {
                    n_tokens: 1,
                    token: token_slice.as_mut_ptr(),
                    embd: ptr::null_mut(),
                    pos: pos.as_mut_ptr(),
                    n_seq_id: ptr::null_mut(),
                    seq_id: ptr::null_mut(),
                    logits: logits_required.as_mut_ptr(),
                };

                let ret = llama_decode(ctx, batch);
                assert_eq!(ret, 0, "Decode failed");

                let logits_ptr = llama_get_logits(ctx);
                assert!(!logits_ptr.is_null(), "Logits pointer is null");

                let vocab_size = llama_vocab_n_tokens(vocab);
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

                llama_sampler_apply(sampler, &mut token_data_array);

                let next_token = llama_sampler_sample(sampler, ctx, 0);
                llama_sampler_accept(sampler, next_token);

                if cur_len < tokens.len() {
                    tokens[cur_len] = next_token;
                    cur_len += 1;
                } else {
                    println!("Token buffer full");
                    break;
                }

                let token_text_ptr = llama_vocab_get_text(vocab, next_token);
                let token_text = std::ffi::CStr::from_ptr(token_text_ptr).to_string_lossy();
                print!("{}", token_text);

                if next_token == 2 || next_token == 0 {
                    break;
                }
            }

            println!();

            llama_sampler_free(sampler);
        // }
        llama_free(ctx);
        llama_free_model(model);
    }
}
// Batch structure: You're only decoding the last token on each step, so:
//
// n_tokens = 1
//
// token = &mut tokens[cur_len - 1]
//
// pos points to current position only
//
// logits = &[true] for that token
//
// Crash fix: The get_logits_ith: invalid logits id 0 came from incorrectly passing n_tokens or reusing the full token array. Now, only the last token is decoded in each loop iteration.
//
// Sampler use:
//
// llama_sampler_apply processes logits
//
// llama_sampler_sample chooses a token
//
// llama_sampler_accept registers the choice
//
// Safe vocab access:
//
// llama_vocab_n_tokens returns vocab size
//
// llama_vocab_get_text retrieves decoded string
//
// End condition: Break on special token (ID 2 or 0)