use anyhow::Result;

// Replace with your own underlying function that sends a prompt to llama.cpp and returns text.
fn llama_run(prompt: &str) -> Result<String> {
    // e.g. crate::llama::infer(prompt) or whatever your repo exposes
    crate::llama::infer(prompt) // <- adjust to your actual function
}

pub fn summarize(text: &str) -> Result<String> {
    let prompt = format!(
        "You are a precise summarizer. Summarize the text in 4-6 bullet points with key facts only.\n\nTEXT:\n{}\n\nSUMMARY:",
        text
    );
    llama_run(&prompt)
}

pub fn answer(question: &str, context: &str) -> Result<String> {
    // If context provided, do extractive + concise. If not, fallback to general knowledge of the model.
    let prompt = if context.trim().is_empty() {
        format!(
            "Answer the question concisely in 2-3 sentences.\n\nQUESTION: {}\n\nANSWER:",
            question
        )
    } else {
        format!(
            "Use ONLY the provided context to answer the question. If the answer isn't in the context, say 'I don't know'. Keep it to 2-4 sentences.

CONTEXT:
{}

QUESTION: {}

ANSWER:", context, question)
    };
    llama_run(&prompt)
}
