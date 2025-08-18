// # REPL chat mode
// cargo run  --bin rusty_llama -- chat
//
// # Generate from file
// cargo run  --bin rusty_llama -- file prompts.txt
//
// # Generate from prompt (multi-word)
// cargo run  --bin rusty_llama -- prompt Explain Rust ownership rules
//
// # CSV query (multi-word query)
// cargo run  --bin rusty_llama -- csv ./data/sales.csv ./output.txt "Given the following CSV data:\n{csv}\n\nCalculate and output only the numeric average sales. Do not provide explanations or additional text. Answer:"

use clap::{Parser};
use std::{ fs, io};

use std::io::Write;
// mod textmod;
mod imagemod;
mod forecastmod;
use rusty_llama::llama::*;
use crate::forecastmod::*;
use crate::imagemod::OrtImageClassifier;


use augurs::ets::{AutoETS, trend::AutoETSTrendModel};
use augurs::mstl::MSTLModel;
use augurs::prelude::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    backend_init();
    remove_logs();

    // cc::Build::new()
    //     .cpp(true)
    //     .file("cpp/image_bridge.cpp")
    //     .file("cpp/forecast_bridge.cpp")
    //     .flag_if_supported("-std=c++17")
    //     .compile("ml_bridges");

    // If/when you link ONNX Runtime or libtorch, add .include() and .cargo:rustc-link-lib here.
    // println!("cargo:rustc-link-search=native=/path/to/onnxruntime/lib");
    // println!("cargo:rustc-link-lib=dylib=onnxruntime");
    let cli = Cli::parse();
    let model_params = model_default_params();
    let model = Model::load(&cli.model, model_params)?;

    let ctx_params = context_default_params();
    let ctx = Context::new(&model, ctx_params)?;

    let vocab = model.get_vocab()?;
    let sampler = Sampler::new(&cli).expect("Failed to create sampler");
//     let model_path_cstr = CString::new(cli.model.clone()).unwrap();
//     println!("model_path_cstr : {:?}", model_path_cstr);
//
//     let model_params = model_default_params();
//     let model = load_model_from_file(model_path_cstr.as_c_str(), model_params)
//         .expect("Failed to load model");
//
//     let ctx_params = context_default_params();
//     let ctx = new_context_with_model(model, ctx_params)
//         .expect("Failed to create context");
//
//     let vocab = model_get_vocab(model)
//         .expect("Failed to get vocab");
//
//     // let model = Model::load(model_path_cstr, params)?;
//
//
//     // let params = model_default_params();
//     // print_model_params(&params);
//
//     // let ctx_params = context_default_params();
//     // print_context_params(&ctx_params);
//     // println!("About to load model at path: {}", cli.model);
//     // let model = load_model_from_file(model_path_cstr.as_c_str(), model_params)
//     //     .expect("Failed to load model");
//     // let model = Model::load(&cli.model, params)?;
//     println!("Model loaded: {:?}", model);
//
//     // let ctx = Context::new(&model, ctx_params)?;
//     // println!("ctx: {:?}", ctx);
//     // let vocab = model_get_vocab(model)
//     //     .expect("Failed to get vocab");
//     // let vocab = model.get_vocab()?;
//     println!("vocab: {:?}", vocab);
//     let sampler = setup_sampler(&cli).expect("Failed to initialize sampler");
//     println!("sampler: {:?}", sampler);
//     // let sampler = Sampler::new(&cli).expect("Failed to create sampler");
    assert!(!sampler.ptr().is_null(), "Failed to init sampler");

    match cli.command {
        Commands::Chat => {
            loop {
                print!("\n> ");
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if input.eq_ignore_ascii_case("exit") {
                    break;
                }

                let output = generate_text(ctx.ptr(), vocab.ptr(), sampler.ptr(), input)
                    .expect("generating answer failed");
                // let output = generate_text(ctx.as_ptr(), vocab, sampler, input);
                println!("{:?}", output);
            }
        }
        Commands::File { filename } => {
            let text = fs::read_to_string(filename).unwrap();
            // let output = generate_text(ctx.as_ptr(), vocab, sampler, &text);
            let output = generate_text(ctx.ptr(), vocab.ptr(), sampler.ptr(), &text)
                .expect("sending file failed");
            println!("{:?}", output);
        }
        Commands::Prompt { text } => {
            let prompt = text.join(" ");
            // let output = generate_text(ctx.as_ptr(), vocab, sampler, &prompt);
            let output = generate_text(ctx.ptr(), vocab.ptr(), sampler.ptr(), &prompt)
                .expect("sending prompt failed");
            println!("{:?}", output);
        }
        Commands::SummarizePrompt { text } => {
            let prompt = text.unwrap_or("".to_string());
            if prompt.trim().is_empty() {
                println!("Prompt can't be empty");
            }
            let summary_prompt = format!(
                "Summarize the following text in a concise manner:\n\n{}\n\nSummary:",
                prompt
            );
            // let output = generate_text(ctx.as_ptr(), vocab, sampler, &prompt);
            let output = generate_text(ctx.ptr(), vocab.ptr(), sampler.ptr(), &summary_prompt)
                .expect("sending prompt failed");
            println!("{:?}", output);
        }
        Commands::SummarizeFile { filename } => {
            let text = fs::read_to_string(filename).unwrap();
            let summary_prompt = format!(
                "Summarize the following text in a concise manner:\n\n{}\n\nSummary:",
                text
            );
            let output = generate_text(ctx.ptr(), vocab.ptr(), sampler.ptr(), &summary_prompt)
                .expect("sending file failed");
            println!("{:?}", output);
        }
        Commands::Csv { csv_path, output_path, query } => {
            let prompt_template = query.join(" ").replace("\\n", "\n");
            run_csv_query(ctx.ptr(), vocab.ptr(), sampler.ptr(), &csv_path, &prompt_template, &output_path)
                // run_csv_query(ctx.as_ptr(), vocab, sampler, &csv_path, &prompt_template, &output_path)
                .expect("CSV query failed");
        }
        Commands::Answer { question, context_file } => {
            let context = if let Some(ref _f) = context_file {
                fs::read_to_string(context_file.clone().unwrap()).unwrap_or("".to_string())
            } else {
                String::new()
            };

            let context_prompt = format!(
                "Answer the following question \n{}\n using the context from the file :\n\n{}\n\n If no file provided, \
                answer the question in a concise manner. Summary:",
                question, context
            );

            let output = generate_text(ctx.ptr(), vocab.ptr(), sampler.ptr(), &context_prompt)
                .expect("sending prompt failed");
            println!("{:?}", output);
        }
        Commands::Classify { model, image , labels } => {
            let classifier = imagemod::OrtImageClassifier::new(&model)?;

            // TEST preprocess image -> tensor
            // let input = ndarray::Array::from_shape_vec((1, 3, 224, 224),
            //                                            vec![0.5; 1*3*224*224]
            // )?;
            let input = imagemod::preprocess_image(&image);

            let outputs = OrtImageClassifier::classify(&classifier, input)?;



            // load labels file
            let label_list: Vec<String> = std::fs::read_to_string(&labels)?
                .lines()
                .map(|s| s.to_string())
                .collect();

            // assume output[0] is probabilities
            // let probs = outputs.view();
            // let probs = outputs;
            // let mut best_idx = 0;
            // let mut best_score = f32::MIN;
            //
            // for (i, score) in probs.iter().enumerate() {
            //     if *score > best_score {
            //         best_score = *score;
            //         best_idx = i;
            //     }
            // }
            // println!("Prediction: {} (score: {:.3})", label_list[best_idx], best_score);


            let probs = imagemod::softmax(&outputs);

            // Find top-3 predictions
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // descending

            println!("Top-3 predictions:");
            for (idx, prob) in indexed_probs.iter().take(3) {
                println!("  {} ({:.3})", label_list[*idx], prob);
            }
            // println!("Prediction: {} (prob: {:.3})", label_list[best_idx], best_prob);

        }
        Commands::Forecast { model, input_data, steps } => {
            // Load CSV -> Vec<f64>
            let values = forecastmod::preprocess_forecast_input(&input_data)?;

            // Fit ETS model
            let model = AutoETS::non_seasonal();
            let fitted = model.fit(&values)?;

            // Forecast `steps` values
            let forecasted = fitted.predict(steps.try_into().unwrap(), 0.95)?; // 95% confidence interval

            // println!("Forecast for next {} steps: {:?}", steps, forecasted);
            println!("Forecast for next {} steps:", steps);
            for (i, val) in forecasted.point.iter().enumerate() {
                println!("Step {}: {:.3}", i + 1, val);
            }
        }
        _ => {}
    }
//
//     sampler_free(sampler);
//     free_context(ctx);
//     free_model(model);
    Ok(())
}

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     backend_init();
//
//     let cli = Cli::parse();
//
//     let model_params = model_default_params();
//     let model = Model::load(&cli.model, model_params)?;
//
//     let ctx_params = context_default_params();
//     let ctx = Context::new(&model, ctx_params)?;
//
//     let vocab = model.get_vocab()?;
//     let sampler = Sampler::new(&cli).expect("Failed to create sampler");
//     // Now pass safe pointers to ffi functions
//     generate_text(ctx.ptr(), vocab.ptr(), sampler.ptr(), "some prompt");
//
//     Ok(())
// }
