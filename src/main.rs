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
}

fn main() {
    unsafe {
        // llama_backend_init();
        unsafe {
            llama_backend_init();
        }

        let path = CString::new("models/ggml-model-q4_0.gguf").unwrap();
        let model_params = llama_model_default_params();
        let model = llama_load_model_from_file(path.as_ptr(), model_params);

        if model.is_null() {
            panic!("Failed to load model.");
        }
        println!("MODEL! {:?}", model.as_ref().to_owned().clone());

        let ctx_params = llama_context_default_params();
        let ctx = llama_new_context_with_model(model, ctx_params);
        if ctx.is_null() {
            llama_free_model(model);
            panic!("Failed to create context.");
        }
        assert!(!ctx.is_null());

        println!("✅ Model and context loaded successfully!");
        // ✅ If you reach here, model + context loaded successfully
        // Get vocab for token-to-text conversion
        // let vocab = llama_model_get_vocab(model);
        // assert!(!vocab.is_null());
        let vocab = unsafe { llama_model_get_vocab(model) };
        println!("{:?}", &vocab);
        assert!(!vocab.is_null(), "Failed to get vocab");

        // Tokenize prompt
        // let prompt = CString::new("Hello world").unwrap();

        let prompt_str = "Hello world";
        let prompt_c = CString::new(prompt_str).unwrap();
        assert!(!ctx.is_null(), "Context pointer is null!");
        let mut tokens = [0i32; 512];
        let n_tokens = llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            prompt_str.len() as i32,
            tokens.as_mut_ptr(),
            512,
            true,  // add_special
            false, // parse_special
        );

        // // Decode (run inference)
        let batch = LlamaBatch {
            n_tokens,
            token: tokens.as_mut_ptr(),
            embd: ptr::null_mut(),
            pos: ptr::null_mut(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: ptr::null_mut(),
        };

//         ⚠️ Common Crashes & Fixes
//         Problem	Symptom	Fix
//         llama_get_logits() before llama_decode()	Segfault or garbage data	Always decode before accessing logits
//         Using moved or dropped token_data	Crash during apply()	Ensure token_data is not modified/moved
//         logits_ptr is null	Segfault at add()	Check logits_ptr.is_null()
//         vocab_size is too large	Panic or crash	Validate llama_n_vocab(model) returns expected size
        // Step 1: Run decode (you must do this first)
        let ret = unsafe { llama_decode(ctx, batch) };
        assert_eq!(ret, 0, "Decoding failed");
        println!("Decode returned: {}", ret);

        let logits_ptr = unsafe { llama_get_logits(ctx) }; // pointer to float32 logits
        assert!(!logits_ptr.is_null(), "Logits pointer is null");

        let vocab_size = unsafe { llama_vocab_n_tokens(vocab) };
        // let vocab_size = unsafe { llama_n_vocab(ctx) };
        println!("Vocab size: {}", vocab_size);
        assert!(!vocab_size.to_string().is_empty(), "vocab_size is is_empty");
        // let vocab_size = 5;
        let mut token_data: Vec<LlamaTokenData> = (0..vocab_size).map(|i| {
            LlamaTokenData {
                id: i as i32,
                logit: unsafe { *logits_ptr.add(i.try_into().unwrap()) },
                p: 0.0,
            }
        }).collect();
        // let mut token_data: Vec<LlamaTokenData> = (0..vocab_size).map(|i| {
        //     LlamaTokenData {
        //         id: i as i32,
        //         logit: unsafe { *logits_ptr.add(i) },
        //         p: 0.0,
        //     }
        // }).collect();
        // println!("{:?}", &token_data.to_vec());

        assert!(!token_data.is_empty(), "Failed to token_data");

        // build token data array
        let mut token_data_array = LlamaTokenDataArray {
            data: token_data.as_mut_ptr(),
            size: token_data.len(),
            sorted: false,
        };
        // assert!(!token_data_array.try_into().unwrap().some(), "Failed to token_data_array");

        let sampler = unsafe { llama_sampler_init_greedy() }; // or top_k, etc.
        assert!(!sampler.is_null(), "Failed to init sampler");
        println!("after sampler");
        println!("Sampler {:?}", &sampler);


        // 4. Apply sampler to logits
        llama_sampler_apply(sampler, &mut token_data_array);
        println!("after llama_sampler_apply");

        // 5. Sample next token
        let next_token = llama_sampler_sample(sampler, ctx, 0); // batch index 0
        println!("after next_token");

        // 6. Accept token in sampler
        llama_sampler_accept(sampler, next_token);
        println!("after llama_sampler_accept");

        // Cleanup
        llama_sampler_free(sampler);
        // Cleanup
        llama_free(ctx);
        llama_free_model(model);
    }
}