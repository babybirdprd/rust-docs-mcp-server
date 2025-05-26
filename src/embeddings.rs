use crate::{
    doc_loader::Document,
    error::ServerError,
    services::{EmbeddingGenerator, Reranker, RerankedDocument},
};
use anyhow::{anyhow, Result, Error as AnyhowError};
use async_openai::{
    config::OpenAIConfig,
    error::ApiError as OpenAIAPIErr,
    types::{CreateEmbeddingRequestArgs, Embedding as OpenAIEmbedding}, // Added Embedding
    Client as OpenAIClient,
};
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use ndarray::{Array1, ArrayView1};
use std::sync::Arc;
use std::sync::OnceLock; // Keep OnceLock if OPENAI_CLIENT static is still used elsewhere.
use tiktoken_rs::cl100k_base;

// Static OnceLock for the OpenAI client - can be used for default client or removed if client is always passed.
// For now, OpenAIEmbeddingClient will store its own client.
pub static OPENAI_CLIENT: OnceLock<OpenAIClient<OpenAIConfig>> = OnceLock::new();

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

// Define a struct containing path, content, and embedding for caching
#[derive(Serialize, Deserialize, Debug, Encode, Decode)]
pub struct CachedDocumentEmbedding {
    pub path: String,
    pub content: String, // Add the extracted document content
    pub vector: Vec<f32>,
}

/// Calculates the cosine similarity between two vectors.
pub fn cosine_similarity(v1: ArrayView1<f32>, v2: ArrayView1<f32>) -> f32 {
    let dot_product = v1.dot(&v2);
    let norm_v1 = v1.dot(&v1).sqrt();
    let norm_v2 = v2.dot(&v2).sqrt();

    if norm_v1 == 0.0 || norm_v2 == 0.0 {
        0.0
    } else {
        dot_product / (norm_v1 * norm_v2)
    }
}

// --- OpenAI Embedding Client ---

pub struct OpenAIEmbeddingClient {
    client: OpenAIClient<OpenAIConfig>,
    model: String, // Model name, e.g., "text-embedding-3-small"
}

impl OpenAIEmbeddingClient {
    pub fn new(client: OpenAIClient<OpenAIConfig>, model: String) -> Self {
        Self { client, model }
    }

    // Helper to make a single embedding request to OpenAI
    async fn fetch_openai_embedding(&self, input: Vec<String>) -> Result<OpenAIEmbedding, AnyhowError> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(input.clone()) // Clone input here
            .build()
            .map_err(|e| anyhow!("Failed to build OpenAI request: {}", e))?;

        let response = self.client.embeddings().create(request).await
            .map_err(|e| anyhow!("OpenAI API call failed: {}", e))?;

        response.data.into_iter().next()
            .ok_or_else(|| anyhow!("OpenAI API returned no embedding data for input: {:?}", input))
    }
}

#[async_trait]
impl EmbeddingGenerator for OpenAIEmbeddingClient {
    async fn generate_embeddings(
        &self,
        documents: Vec<String>,
    ) -> Result<(Vec<Vec<f32>>, usize), AnyhowError> { // Updated return type
        eprintln!(
            "Generating embeddings for {} documents using OpenAI model {}...",
            documents.len(),
            self.model
        );

        let bpe = Arc::new(
            cl100k_base().map_err(|e| anyhow!("Failed to load BPE tokenizer: {}", e))?,
        );

        const CONCURRENCY_LIMIT: usize = 8;
        const TOKEN_LIMIT: usize = 8000; // OpenAI's limit for text-embedding-3-small is 8191

        let results: Vec<Result<Option<Vec<f32>>, AnyhowError>> = stream::iter(documents)
            .enumerate()
            .map(|(index, doc_content)| {
                let client = self.client.clone(); // Arc clone
                let model_name = self.model.clone(); // Arc clone
                let bpe_clone = Arc::clone(&bpe);

                async move {
                    let token_count = bpe_clone.encode_with_special_tokens(&doc_content).len();

                    if token_count > TOKEN_LIMIT {
                        eprintln!(
                            "    Skipping document {}: Token count ({}) exceeds limit ({}).",
                            index + 1,
                            token_count,
                            TOKEN_LIMIT
                        );
                        return Ok(None); // Skip this document
                    }

                    let request_args = CreateEmbeddingRequestArgs::default()
                        .model(&model_name)
                        .input(vec![doc_content.clone()]) // API expects Vec<String>
                        .build()
                        .map_err(|e| anyhow!("Failed to build OpenAI request for doc {}: {}", index + 1, e))?;
                    
                    match client.embeddings().create(request_args).await {
                        Ok(response) => {
                            let tokens = response.usage.total_tokens as usize;
                            if let Some(embedding_data) = response.data.first() {
                                Ok(Some((embedding_data.embedding.clone(), tokens))) // Return embedding and tokens
                            } else {
                                Err(anyhow!("OpenAI API returned no embedding for doc {}", index + 1))
                            }
                        }
                        Err(e) => Err(anyhow!("OpenAI API call failed for doc {}: {}", index + 1, e)),
                    }
                }
            })
            .buffer_unordered(CONCURRENCY_LIMIT)
            .collect()
            .await;

        let mut final_embeddings = Vec::new();
        let mut total_processed_tokens = 0;
        let mut successful_count = 0;

        for res in results {
            match res {
                Ok(Some((embedding, tokens))) => {
                    final_embeddings.push(embedding);
                    total_processed_tokens += tokens;
                    successful_count += 1;
                }
                Ok(None) => { /* Document was skipped */ }
                Err(e) => {
                    eprintln!("Error processing a document with OpenAI: {}", e);
                    // Optionally: return Err(e) to fail the whole batch.
                }
            }
        }
        
        eprintln!(
            "Finished generating embeddings with OpenAI. Successfully processed {} out of {} documents. Total tokens: {}.",
            successful_count,
            documents.len(),
            total_processed_tokens
        );

        if successful_count == 0 && !documents.is_empty() {
            return Err(anyhow!("No documents were successfully embedded by OpenAI. Check logs for errors."));
        }

        Ok((final_embeddings, total_processed_tokens)) // Return embeddings and total tokens
    }

    async fn generate_single_embedding(
        &self,
        document: String,
    ) -> Result<Vec<f32>, AnyhowError> {
        eprintln!(
            "Generating single embedding using OpenAI model {}...",
            self.model
        );
        // This helper is also used by generate_embeddings, ensure it's compatible or adjust.
        // For single embedding, token count is part of the main response.
        eprintln!(
            "Generating single embedding using OpenAI model {} for document (first 50 chars): {}...",
            self.model, document.chars().take(50).collect::<String>()
        );

        let bpe = cl100k_base().map_err(|e| anyhow!("Failed to load BPE tokenizer for single embedding: {}", e))?;
        let token_count_estimate = bpe.encode_with_special_tokens(&document).len();
        const TOKEN_LIMIT: usize = 8000; 

        if token_count_estimate > TOKEN_LIMIT {
            return Err(anyhow!(
                "Document token count ({}) for single embedding exceeds limit ({}).",
                token_count_estimate, TOKEN_LIMIT
            ));
        }
        
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(vec![document.clone()])
            .build()
            .map_err(|e| anyhow!("Failed to build OpenAI request for single embedding: {}", e))?;

        let response = self.client.embeddings().create(request).await
            .map_err(|e| anyhow!("OpenAI API call failed for single embedding: {}", e))?;

        // We don't need to return token count for single embedding per trait, but good to log.
        // eprintln!("Tokens used for single embedding: {}", response.usage.total_tokens);

        response.data.into_iter().next()
            .map(|data| data.embedding)
            .ok_or_else(|| anyhow!("OpenAI API returned no embedding for single input document"))
    }
}

// --- OpenAI Reranker ---

pub struct OpenAIReranker; // Empty struct for now

#[async_trait]
impl Reranker for OpenAIReranker {
    async fn rerank_documents(
        &self,
        _query: String, // Query is unused in this placeholder
        documents: Vec<String>,
    ) -> Result<Vec<RerankedDocument>, AnyhowError> {
        eprintln!("OpenAIReranker: Returning documents in original order (placeholder).");
        let reranked_docs = documents
            .into_iter()
            .enumerate()
            .map(|(index, text)| RerankedDocument {
                index,
                text,
                score: 1.0, // Dummy score
            })
            .collect();
        Ok(reranked_docs)
    }
}


// The original generate_embeddings function that worked with Vec<Document>
// and returned (Vec<(String, Array1<f32>)>, usize) might be useful
// to keep for the existing main.rs logic that expects paths and Array1<f32>.
// Or, main.rs needs to be updated to use the new service traits.
// For now, I'll keep it and mark it as potentially deprecated or for internal use.

/// Generates embeddings for a list of `Document` structs using the OpenAI API.
/// This version is kept for compatibility with existing logic in main.rs that
/// expects (String path, Array1<f32> embedding) and token counts.
/// It could be refactored to use OpenAIEmbeddingClient internally if desired.
pub async fn generate_embeddings_with_paths_and_stats(
    client: &OpenAIClient<OpenAIConfig>, // Accepts a client
    documents: &[Document],
    model: &str,
) -> Result<(Vec<(String, Array1<f32>)>, usize), ServerError> {
    let bpe = Arc::new(cl100k_base().map_err(|e| ServerError::Tiktoken(e.to_string()))?);
    const CONCURRENCY_LIMIT: usize = 8;
    const TOKEN_LIMIT: usize = 8000;

    let results = stream::iter(documents.iter().enumerate())
        .map(|(index, doc)| {
            let client = client.clone();
            let model_name = model.to_string(); // Use model_name to avoid conflict
            let doc_clone = doc.clone(); // Use doc_clone
            let bpe_clone = Arc::clone(&bpe);

            async move {
                let token_count = bpe_clone.encode_with_special_tokens(&doc_clone.content).len();
                if token_count > TOKEN_LIMIT {
                    return Ok(None); // Skipped
                }

                let request = CreateEmbeddingRequestArgs::default()
                    .model(&model_name)
                    .input(vec![doc_clone.content.clone()])
                    .build()
                    .map_err(|e| ServerError::OpenAI(async_openai::error::OpenAIError::InvalidArgument(e.to_string())))?;
                
                match client.embeddings().create(request).await {
                    Ok(response) => {
                        if let Some(embedding_data) = response.data.first() {
                            Ok(Some((
                                doc_clone.path.clone(),
                                Array1::from(embedding_data.embedding.clone()),
                                token_count,
                            )))
                        } else {
                            Err(ServerError::OpenAI(async_openai::error::OpenAIError::ApiError(OpenAIAPIErr {
                                message: format!("No embedding data for doc {}", doc_clone.path),
                                r#type: Some("sdk_error".to_string()), param: None, code: None,
                            })))
                        }
                    }
                    Err(e) => Err(ServerError::OpenAI(e)),
                }
            }
        })
        .buffer_unordered(CONCURRENCY_LIMIT)
        .collect::<Vec<Result<Option<(String, Array1<f32>, usize)>, ServerError>>>()
        .await;

    let mut embeddings_vec = Vec::new();
    let mut total_processed_tokens = 0;
    for result in results {
        match result {
            Ok(Some((path, embedding, tokens))) => {
                embeddings_vec.push((path, embedding));
                total_processed_tokens += tokens;
            }
            Ok(None) => {} // Document was skipped
            Err(e) => {
                eprintln!("Error during OpenAI embedding (with paths): {}", e);
                // Potentially return error or continue
                return Err(e); 
            }
        }
    }
    Ok((embeddings_vec, total_processed_tokens))
}