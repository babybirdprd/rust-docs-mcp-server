// src/voyage_client.rs

use crate::services::{EmbeddingGenerator, Reranker, RerankedDocument};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

// Common Usage struct for both Embedding and Rerank APIs
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoyageUsage {
    pub total_tokens: i32,
}

// Structs for the /embeddings endpoint

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)] // Allows input to be String or Vec<String>
pub enum VoyageInput {
    String(String),
    StringArray(Vec<String>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoyageEmbeddingRequest {
    pub input: VoyageInput,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dtype: Option<String>, // Not used for now, but part of spec
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>, // e.g., "base64"
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoyageEmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>, // Assuming Vec<f32> for now as per instructions
    pub index: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoyageEmbeddingResponse {
    pub object: String,
    pub data: Vec<VoyageEmbeddingData>,
    pub model: String,
    pub usage: VoyageUsage,
}

// Structs for the /rerank endpoint

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoyageRerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoyageRerankData {
    pub index: i32,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoyageRerankResponse {
    pub object: String,
    pub data: Vec<VoyageRerankData>,
    pub model: String,
    pub usage: VoyageUsage,
}

const VOYAGE_API_BASE_URL: &str = "https://api.voyageai.com/v1";

pub async fn create_voyage_embeddings(
    base_url: &str, // New parameter
    request_body: &VoyageEmbeddingRequest,
    api_key: &str,
    client: &Client,
) -> Result<VoyageEmbeddingResponse, reqwest::Error> {
    let url = format!("{}/embeddings", base_url); // Use base_url

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(request_body)
        .send()
        .await?;

    response
        .json::<VoyageEmbeddingResponse>()
        .await
        .context("Failed to deserialize VoyageEmbeddingResponse")
}

pub async fn rerank_voyage_documents(
    base_url: &str, // New parameter
    request_body: &VoyageRerankRequest,
    api_key: &str,
    client: &Client,
) -> Result<VoyageRerankResponse, reqwest::Error> {
    let url = format!("{}/rerank", base_url); // Use base_url

    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(request_body)
        .send()
        .await
        .context("HTTP request for Voyage rerank failed")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
        return Err(anyhow!("Voyage rerank API request failed with status {}: {}", status, text).into());
    }
    
    response
        .json::<VoyageRerankResponse>()
        .await
        .context("Failed to deserialize VoyageRerankResponse")
}

// --- Voyage AI Client implementing traits ---

pub struct VoyageAIClient {
    client: Client,
    api_key: String,
    embedding_model: String,
    rerank_model: String,
    base_url: String, // New field for testability
}

impl VoyageAIClient {
    pub fn new(
        client: Client,
        api_key: String,
        embedding_model: String,
        rerank_model: String,
        base_url: Option<String>, // Optional for constructor
    ) -> Self {
        Self {
            client,
            api_key,
            embedding_model,
            rerank_model,
            base_url: base_url.unwrap_or_else(|| VOYAGE_API_BASE_URL.to_string()),
        }
    }
}

#[async_trait]
impl EmbeddingGenerator for VoyageAIClient {
    async fn generate_embeddings(
        &self,
        documents: Vec<String>,
    ) -> Result<(Vec<Vec<f32>>, usize), anyhow::Error> {
        if documents.is_empty() {
            return Ok((Vec::new(), 0));
        }
        // eprintln!( // Comment out noisy eprintln for tests
        //     "Generating embeddings for {} documents using Voyage AI model {}...",
        //     documents.len(),
        //     self.embedding_model
        // );

        let request_body = VoyageEmbeddingRequest {
            input: VoyageInput::StringArray(documents),
            model: self.embedding_model.clone(),
            input_type: None,
            truncation: Some(true),
            output_dimension: None,
            output_dtype: None,
            encoding_format: None,
        };

        let response = create_voyage_embeddings(
            &self.base_url, // Use the base_url from the struct
            &request_body,
            &self.api_key,
            &self.client,
        )
        .await
        .map_err(|e| anyhow!("Voyage embedding API call failed: {}", e))?;

        let embeddings = response
            .data
            .into_iter()
            .map(|data| data.embedding)
            .collect();
        
        let total_tokens = response.usage.total_tokens as usize;
        eprintln!(
            "Finished generating embeddings with Voyage AI. Processed {} documents. Total tokens: {}.",
            embeddings.len(), // This should be same as documents.len() if no errors before this point
            total_tokens
        );
        Ok((embeddings, total_tokens)) // Return embeddings and total tokens
    }

    async fn generate_single_embedding(
        &self,
        document: String,
    ) -> Result<Vec<f32>, anyhow::Error> {
        // eprintln!( // Comment out noisy eprintln for tests
        //     "Generating single embedding using Voyage AI model {}...",
        //     self.embedding_model
        // );
        let request_body = VoyageEmbeddingRequest {
            input: VoyageInput::String(document),
            model: self.embedding_model.clone(),
            input_type: None,
            truncation: Some(true),
            output_dimension: None,
            output_dtype: None,
            encoding_format: None,
        };

        let response = create_voyage_embeddings(
            &self.base_url, // Use the base_url from the struct
            &request_body,
            &self.api_key,
            &self.client,
        )
        .await
        .map_err(|e| anyhow!("Voyage single embedding API call failed: {}", e))?;

        response
            .data
            .into_iter()
            .next()
            .map(|data| data.embedding)
            .ok_or_else(|| anyhow!("Voyage API returned no embedding for single input"))
    }
}

#[async_trait]
impl Reranker for VoyageAIClient {
    async fn rerank_documents(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> Result<Vec<RerankedDocument>, anyhow::Error> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        // eprintln!( // Comment out noisy eprintln for tests
        //     "Reranking {} documents using Voyage AI model {}...",
        //     documents.len(),
        //     self.rerank_model
        // );

        let request_body = VoyageRerankRequest {
            query,
            documents: documents.clone(),
            model: self.rerank_model.clone(),
            top_k: None,
            return_documents: Some(true),
            truncation: Some(true),
        };

        let response = rerank_voyage_documents(
            &self.base_url, // Use the base_url from the struct
            &request_body,
            &self.api_key,
            &self.client,
        )
        .await
        .map_err(|e| anyhow!("Voyage rerank API call failed: {}", e))?;

        // The Voyage API returns results sorted by relevance.
        // The `index` in `VoyageRerankData` refers to the original index in the `documents` input array.
        let reranked_docs = response
            .data
            .into_iter()
            .map(|data| {
                let original_doc_text = data.document.or_else(|| {
                    // Fallback if document is not returned, though we requested it.
                    // This requires `documents` to be captured or available.
                    // The `data.index` is the original index.
                    documents.get(data.index as usize).cloned()
                }).unwrap_or_else(|| {
                    eprintln!("Warning: Document text missing for index {} in rerank response.", data.index);
                    String::new() // Or handle as an error
                });

                RerankedDocument {
                    index: data.index as usize, // Original index from the input list
                    text: original_doc_text,
                    score: data.relevance_score,
                }
            })
            .collect();

        Ok(reranked_docs)
    }
}


// Example of how to instantiate VoyageInput
#[cfg(test)]
mod tests {
    use super::*;
    use mockito::{Server, Mock, ServerGuard}; // Updated import for mockito 1.x
    use reqwest::Client;
    use serde_json::json; // For creating JSON bodies easily

    // Helper to create a common reqwest client for tests
    fn test_client() -> Client {
        Client::new()
    }

    #[tokio::test]
    async fn test_create_voyage_embeddings_success() {
        let mut server = Server::new_async().await; // Use new_async for mockito 1.x
        let mock_url = server.url();

        let mock_api_key = "test_api_key";

        let request_payload = VoyageEmbeddingRequest {
            input: VoyageInput::StringArray(vec!["doc1".to_string(), "doc2".to_string()]),
            model: "voyage-2".to_string(),
            input_type: None,
            truncation: None,
            output_dimension: None,
            output_dtype: None,
            encoding_format: None,
        };

        let expected_response_body = json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                },
                {
                    "object": "embedding",
                    "embedding": [0.4, 0.5, 0.6],
                    "index": 1
                }
            ],
            "model": "voyage-2",
            "usage": {
                "total_tokens": 10
            }
        });

        let _m: Mock = server.mock("POST", "/embeddings") // Mock is bound to server lifetime
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_header("authorization", &format!("Bearer {}", mock_api_key))
            .with_json_body(&request_payload) // mockito serializes this
            .with_body(expected_response_body.to_string()) // Provide raw JSON string
            .create_async()
            .await;

        let client = test_client();
        let result = create_voyage_embeddings(
            &mock_url,
            &request_payload,
            mock_api_key,
            &client,
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.model, "voyage-2");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(response.usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn test_create_voyage_embeddings_error() {
        let mut server = Server::new_async().await;
        let mock_url = server.url();
        let mock_api_key = "test_api_key_error";

        let request_payload = VoyageEmbeddingRequest {
            input: VoyageInput::String("error case".to_string()),
            model: "voyage-2".to_string(),
            input_type: None, truncation: None, output_dimension: None, output_dtype: None, encoding_format: None,
        };

        let _m = server.mock("POST", "/embeddings")
            .with_status(500)
            .with_header("authorization", &format!("Bearer {}", mock_api_key))
            .with_json_body(&request_payload)
            .with_body("Internal Server Error")
            .create_async()
            .await;

        let client = test_client();
        let result = create_voyage_embeddings(
            &mock_url,
            &request_payload,
            mock_api_key,
            &client,
        )
        .await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_rerank_voyage_documents_success() {
        let mut server = Server::new_async().await;
        let mock_url = server.url();
        let mock_api_key = "test_rerank_key";

        let request_payload = VoyageRerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc A".to_string(), "doc B".to_string()],
            model: "rerank-lite-1".to_string(),
            top_k: None,
            return_documents: Some(true),
            truncation: None,
        };

        let expected_response_body = json!({
            "object": "list",
            "data": [
                { "index": 1, "relevance_score": 0.95, "document": "doc B" },
                { "index": 0, "relevance_score": 0.85, "document": "doc A" }
            ],
            "model": "rerank-lite-1",
            "usage": { "total_tokens": 20 }
        });

        let _m = server.mock("POST", "/rerank")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_header("authorization", &format!("Bearer {}", mock_api_key))
            .with_json_body(&request_payload)
            .with_body(expected_response_body.to_string())
            .create_async()
            .await;

        let client = test_client();
        let result = rerank_voyage_documents(
            &mock_url,
            &request_payload,
            mock_api_key,
            &client,
        )
        .await;

        assert!(result.is_ok(), "Expected Ok, got Err: {:?}", result.err());
        let response = result.unwrap();
        assert_eq!(response.model, "rerank-lite-1");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].relevance_score, 0.95);
        assert_eq!(response.data[0].document, Some("doc B".to_string()));
        assert_eq!(response.usage.total_tokens, 20);
    }

    #[tokio::test]
    async fn test_rerank_voyage_documents_error() {
        let mut server = Server::new_async().await;
        let mock_url = server.url();
        let mock_api_key = "test_rerank_error_key";

        let request_payload = VoyageRerankRequest {
            query: "error query".to_string(),
            documents: vec!["doc X".to_string()],
            model: "rerank-lite-1".to_string(),
            top_k: None, return_documents: None, truncation: None,
        };

        let _m = server.mock("POST", "/rerank")
            .with_status(400) // Bad Request
            .with_header("authorization", &format!("Bearer {}", mock_api_key))
            .with_json_body(&request_payload)
            .with_body("Bad Request Payload")
            .create_async()
            .await;
        
        let client = test_client();
        let result = rerank_voyage_documents(
            &mock_url,
            &request_payload,
            mock_api_key,
            &client,
        )
        .await;

        assert!(result.is_err());
        // Optionally, check the error message content if it's specific
        if let Err(e) = result {
            assert!(e.to_string().contains("Voyage rerank API request failed with status 400 Bad Request"));
        }
    }
    
    // Test for serialization of VoyageInput remains valuable.
    #[test]
    fn test_voyage_input_serialization() {
        let single_input = VoyageInput::String("test query".to_string());
        let single_json = serde_json::to_string(&single_input).unwrap();
        assert_eq!(single_json, "\"test query\"");

        let array_input = VoyageInput::StringArray(vec!["doc1".to_string(), "doc2".to_string()]);
        let array_json = serde_json::to_string(&array_input).unwrap();
        assert_eq!(array_json, "[\"doc1\",\"doc2\"]");
    }
}
