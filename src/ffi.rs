use std::ffi::{CString, CStr};
use std::os::raw::{c_int, c_float, c_char, c_void};

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
#[derive(Copy, Clone, Debug)]
pub struct LlamaSamplerChainParams {
    // _private: [u8; SIZE], // ???
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LlamaTokenData {
    pub id: i32,     // llama_token is typedef'd to int32_t
    pub logit: f32,
    pub p: f32,
}

#[repr(C)]
pub struct LlamaMemory {
    pub _private: [u8; 0], // opaque struct
}

#[link(name = "llama")]
extern "C" {
    pub fn llama_backend_init();
    pub fn llama_model_default_params() -> LlamaModelParams;
    pub fn llama_context_default_params() -> LlamaContextParams;
    pub fn llama_load_model_from_file(path: *const c_char, params: LlamaModelParams) -> *mut LlamaModel;
    pub fn llama_new_context_with_model(model: *mut LlamaModel, params: LlamaContextParams) -> *mut LlamaContext;
    pub fn llama_free(ctx: *mut LlamaContext);
    pub fn llama_free_model(model: *mut LlamaModel);

    // Instead of llama_eval
    pub fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> c_int;

    pub fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    pub fn llama_vocab_get_text(vocab: *const LlamaVocab, token: i32) -> *const c_char;

    pub fn llama_sampler_init(
        iface: *const LlamaSamplerI,
        ctx: *mut LlamaContext,
    ) -> *mut LlamaSampler;
    pub fn llama_tokenize(
        vocab: *const LlamaVocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut i32,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    // Sampler functions (adjust names)
    pub fn llama_sampler_free(sampler: *mut LlamaSampler);
    // pub fn llama_sampler_sample(sampler: *mut LlamaSampler, ctx: *mut LlamaContext, idx: i32) -> i32;
    pub fn llama_sampler_sample(
        sampler: *mut LlamaSampler,
        ctx: *mut LlamaContext,
        idx: i32,
    ) -> LlamaToken;
    pub fn llama_sampler_accept(sampler: *mut LlamaSampler, token: i32);
    pub fn llama_sampler_apply(sampler: *mut LlamaSampler, logits: *mut LlamaTokenDataArray);

    pub fn llama_sampler_init_greedy() -> *mut LlamaSampler;
    pub fn llama_sample(sampler: *mut LlamaSampler, ctx: *mut LlamaContext);
    pub fn llama_sampler_accept_token(sampler: *mut LlamaSampler, ctx: *mut LlamaContext) -> i32;

    pub fn llama_get_logits(ctx: *mut LlamaContext) -> *const f32;
    // fn llama_n_vocab(ctx: *const LlamaContext) -> i32;
    pub fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> i32;

    // fn llama_sample_init_greedy(ctx: *mut LlamaContext, logits: *const f32) -> i32;

    // Instead of llama_token_to_str
    pub fn llama_token_get_text(ctx: *const LlamaContext, token: i32) -> *const c_char;

    // fn llama_sampler_create(params: *const SamplerParams) -> *mut LlamaSampler;
    // pub fn llama_get_memory(ctx: *mut LlamaContext) -> *mut llama_memory;
    // DEPRECATED(LLAMA_API struct llama_kv_cache * llama_get_kv_self(struct llama_context * ctx), "use llama_get_memory instead");
    // pub fn llama_get_memory(ctx: *const LlamaContext) -> llama_memory_t;
    pub fn llama_get_memory(ctx: *const LlamaContext) -> *mut LlamaMemory;
    // LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
    // pub type llama_memory_t;
    // Clear the llama memory; data=true clears buffers, false clears only metadata
    pub fn llama_memory_clear(mem: *mut LlamaMemory, data: bool);
    // LLAMA_API void llama_memory_clear(
    // llama_memory_t mem,
    // bool data);
    // fn llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32) -> *mut LlamaSampler;
    pub fn llama_sampler_init_temp(t: f32) -> *mut LlamaSampler;
    pub fn llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32) -> *mut LlamaSampler;

    // LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    pub fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    pub fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    pub fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);

    pub fn llama_sampler_init_top_k(k: i32) -> *mut LlamaSampler;
    pub fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut LlamaSampler;

    pub fn llama_sampler_init_dist(seed: u32) -> *mut LlamaSampler;

    // silence logs
    pub fn llama_log_set(
        callback: Option<extern "C" fn(i32, *const std::os::raw::c_char, *mut c_void)>,
        user_data: *mut c_void,
    );
    // pub fn silent_log_callback(_level: i32, _text: *const std::os::raw::c_char, _user_data: *mut c_void);
}