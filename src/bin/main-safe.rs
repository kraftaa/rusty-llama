use rusty_llama::ffi::{LlamaModelParams, LlamaContextParams};
// use rusty_llama::llama::{Model, Context};
use clap::{Parser};
use std::{ fs, io};
use std::ffi::{CString, CStr};

use std::io::Write;
// use rusty_llama::llama::*;
// { generate_text, setup_sampler, Cli, backend_init, llama_sampler_free,llama_free_model, llama_free,
//                           llama_model_get_vocab, llama_model_default_params, load_model_from_file,
//                           llama_context_default_params, llama_new_context_with_model };

// use rusty_llama::ffi::{ llama_model_get_vocab, llama_model_default_params, llama_load_model_from_file,
//                         llama_context_default_params, llama_new_context_with_model };
// use crate::ffi::{llama_model_params, llama_context_params};
// use crate::llama::{Model, Context};
use rusty_llama::ffi::*;
use rusty_llama::llama::*;
fn main() {
    backend_init();

    let cli = Cli::parse();

    println!("Temperature: {}", cli.temperature);
    println!("Top-k: {}", cli.top_k);
    println!("Top-p: {}", cli.top_p);
    let model_path_cstr = CString::new(cli.model.clone()).unwrap();
    println!("model path {:?}", model_path_cstr);

    let model_params = model_default_params();
    // let model = load_model_from_file(model_path_cstr.as_ptr(), model_params).unwrap();;
    let model = load_model_from_file(model_path_cstr.as_c_str(), model_params).unwrap();;
    assert!(!model.is_null(), "Failed to load model");

    let ctx_params = context_default_params();
    let ctx = new_context_with_model(model, ctx_params);
    assert!(!ctx.clone().expect("REASON").is_null(), "Failed to create context");

    let vocab = model_get_vocab(model);
    assert!(!vocab.clone().expect("REASON").is_null(), "Failed to get vocab");

    let sampler = setup_sampler(&cli);
    assert!(!sampler.is_null(), "Failed to init sampler");

    match cli.command {
        // Commands::Chat { brief } => {
        Commands::Chat => {
            // REPL chat mode
            loop {
                print!("\n> ");
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if input == "exit" || input.eq_ignore_ascii_case("exit"){
                    break;
                }

                // let final_input = if brief {
                //     format!("Provide ONLY the direct answer. No explanations.\n\n{}", input)
                // } else {
                //     input.to_string()
                // };

                // generate_text(ctx, vocab, sampler, input);
                generate_text(ctx.clone().expect("REASON"), vocab.clone().expect("REASON"), sampler, input);
            }
        }
        Commands::File { filename } => {
            let text = fs::read_to_string(filename).unwrap();
            generate_text(ctx.clone().expect("REASON"), vocab.expect("REASON"), sampler, &text);
        }
        Commands::Prompt { text } => {
            let prompt = text.join(" ");
            generate_text(ctx.clone().expect("REASON"), vocab.expect("REASON"), sampler, &prompt);
        }
        Commands::Csv {
            csv_path,
            output_path,
            query,
        } => {
            let prompt_template = query.join(" ").replace("\\n", "\n");

            run_csv_query(ctx.clone().expect("REASON"), vocab.expect("REASON"), sampler, &csv_path, &prompt_template, &output_path).unwrap();
        }
    }

    sampler_free(sampler);
    free_context(ctx.unwrap());
    free_model(model);
}


    // let model_params = llama_model_params {
    //     n_gpu_layers: 1,
    //     // fill in defaults
    // };
    // let context_params = llama_context_params {
    //     seed: 1234,
    //     // fill in defaults
    // };
    //
    // let model = Model::load("models/llama-7b.gguf", model_params)
    //     .expect("Failed to load model");
    //
    // let ctx = Context::new(&model, context_params)
    //     .expect("Failed to create context");
    //
    // println!("Model and context loaded successfully!");
// }

