// src/services.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Represents a document after being reranked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankedDocument {
    pub index: usize,    // Original index from the input list
    pub text: String,    // The document content
    pub score: f32,      // Reranking score
}

/// Trait for services that can generate embeddings.
#[async_trait]
pub trait EmbeddingGenerator {
    /// Generates embeddings for a list of document contents.
    ///
    /// # Arguments
    /// * `documents`: A vector of strings, where each string is the content of a document.
    ///
    /// # Returns
    /// A `Result` containing a vector of embeddings (each embedding is a `Vec<f32>`),
    /// or an `anyhow::Error` if the operation fails.
    async fn generate_embeddings(
        &self,
        documents: Vec<String>,
    ) -> Result<(Vec<Vec<f32>>, usize), anyhow::Error>; // Returns (embeddings, total_tokens)

    /// Generates an embedding for a single document content.
    ///
    /// # Arguments
    /// * `document`: A string representing the content of the document.
    ///
    /// # Returns
    /// A `Result` containing the embedding (`Vec<f32>`),
    /// or an `anyhow::Error` if the operation fails.
    async fn generate_single_embedding(
        &self,
        document: String,
    ) -> Result<Vec<f32>, anyhow::Error>;

    // Potentially add a method to get the model name or dimensions if needed later.
    // fn get_model_name(&self) -> String;
    // fn get_embedding_dimension(&self) -> usize;
}

/// Trait for services that can rerank documents based on a query.
#[async_trait]
pub trait Reranker {
    /// Reranks a list of documents based on a query.
    ///
    /// # Arguments
    /// * `query`: The query string.
    /// * `documents`: A vector of strings, where each string is the content of a document to be reranked.
    ///
    /// # Returns
    /// A `Result` containing a vector of `RerankedDocument` structs, sorted by relevance,
    /// or an `anyhow::Error` if the operation fails.
    async fn rerank_documents(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> Result<Vec<RerankedDocument>, anyhow::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        embeddings::{OpenAIEmbeddingClient, OpenAIReranker},
        voyage_client::{
            VoyageAIClient, VoyageEmbeddingRequest, VoyageEmbeddingResponse, VoyageInput,
            VoyageRerankRequest, VoyageRerankResponse, VoyageUsage, VoyageEmbeddingData, VoyageRerankData,
            create_voyage_embeddings, rerank_voyage_documents, // These are needed for VoyageAIClient tests if testing indirectly
        },
    };
    use async_openai::{
        Client as OpenAIClientSdk, // Renamed for clarity
        config::OpenAIConfig,
        types::{
            CreateEmbeddingResponse, Embedding as OpenAIEmbedding, Usage as OpenAIUsage,
            CreateEmbeddingRequestArgs, // For request construction in tests
            // For Reranking with OpenAI, if we were to mock it (not applicable for placeholder)
        },
    };
    use reqwest::Client as ReqwestClient;
    use tokio;
    use mockito::{Server as MockServer, Mock, ServerGuard as MockServerGuard}; // For mockito 1.x
    use serde_json::json;
    use std::sync::Arc; // If any service is Arc-wrapped, though not directly here for instantiation

    // Helper to create a reqwest client for Voyage tests
    fn test_reqwest_client() -> ReqwestClient {
        ReqwestClient::new()
    }

    // --- OpenAIEmbeddingClient Tests ---

    // Mocking async_openai::Client directly is hard without a library like `mockall`.
    // We'll try to use mockito by configuring the OpenAI client with a mock server URL.
    // This tests if the request is formed correctly and if the response is parsed.

    #[tokio::test]
    async fn test_openai_generate_single_embedding_success() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();

        let config = OpenAIConfig::new().with_api_base(&mock_url);
        let sdk_client = OpenAIClientSdk::with_config(config);
        let model_name = "text-embedding-test".to_string();
        let client = OpenAIEmbeddingClient::new(sdk_client, model_name.clone());

        let document = "This is a test document.".to_string();
        
        let expected_sdk_response = CreateEmbeddingResponse {
            object: "list".to_string(),
            model: model_name.clone(),
            data: vec![OpenAIEmbedding {
                object: "embedding".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            usage: OpenAIUsage {
                prompt_tokens: 5, // Example value
                total_tokens: 5,  // Example value
            },
        };

        let _m: Mock = server.mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            // .with_json_body(&json!({ // This needs to match CreateEmbeddingRequest structure
            //     "input": document.clone(),
            //     "model": model_name.clone(),
            // }))
            .with_body(serde_json::to_string(&expected_sdk_response).unwrap())
            .create_async()
            .await;

        let result = client.generate_single_embedding(document).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_openai_generate_single_embedding_error() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();

        let config = OpenAIConfig::new().with_api_base(&mock_url);
        let sdk_client = OpenAIClientSdk::with_config(config);
        let client = OpenAIEmbeddingClient::new(sdk_client, "text-embedding-test".to_string());

        let document = "Error document".to_string();

        let _m: Mock = server.mock("POST", "/embeddings")
            .with_status(500)
            .create_async()
            .await;

        let result = client.generate_single_embedding(document).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_openai_generate_embeddings_success() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();

        let config = OpenAIConfig::new().with_api_base(&mock_url);
        let sdk_client = OpenAIClientSdk::with_config(config);
        let model_name = "text-embedding-test".to_string();
        let client = OpenAIEmbeddingClient::new(sdk_client, model_name.clone());

        let documents = vec!["doc1".to_string(), "doc2".to_string()];
        
        // Mock two responses, one for each document, as the client iterates
        // This mocking strategy might be tricky if the client batches.
        // The current OpenAIEmbeddingClient sends one request per doc in a stream.
        
        let response1 = CreateEmbeddingResponse {
            object: "list".to_string(), model: model_name.clone(),
            data: vec![OpenAIEmbedding { object: "embedding".to_string(), embedding: vec![0.1], index: 0 }],
            usage: OpenAIUsage { prompt_tokens: 1, total_tokens: 1 },
        };
        let response2 = CreateEmbeddingResponse {
            object: "list".to_string(), model: model_name.clone(),
            data: vec![OpenAIEmbedding { object: "embedding".to_string(), embedding: vec![0.2], index: 0 }], // OpenAI API returns index 0 for single input
            usage: OpenAIUsage { prompt_tokens: 1, total_tokens: 1 },
        };

        // This needs careful handling of multiple mock calls if they are sequential and distinct.
        // Mockito handles this by matching requests. If requests are identical, it might be an issue.
        // Assuming inputs are distinct enough for separate mocks if needed, or one mock handles all.
        // For simplicity, we'll assume the mock server can distinguish or the client sends distinct enough requests.
        // The current implementation sends separate identical requests for each doc in the stream.
        // So, we need two mocks that can be consumed.
        server.mock("POST", "/embeddings")
            // .match_body(Matcher::Json(json!({"input": "doc1", "model": model_name})))
            .with_status(200)
            .with_body(serde_json::to_string(&response1).unwrap())
            .create_async().await; // Mock for first call
        server.mock("POST", "/embeddings")
            // .match_body(Matcher::Json(json!({"input": "doc2", "model": model_name})))
            .with_status(200)
            .with_body(serde_json::to_string(&response2).unwrap())
            .create_async().await; // Mock for second call


        let result = client.generate_embeddings(documents).await;
        assert!(result.is_ok());
        let (embeddings, total_tokens) = result.unwrap();
        assert_eq!(embeddings, vec![vec![0.1], vec![0.2]]);
        assert_eq!(total_tokens, 2); // 1 token per doc
    }

    #[tokio::test]
    async fn test_openai_generate_embeddings_error() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();

        let config = OpenAIConfig::new().with_api_base(&mock_url);
        let sdk_client = OpenAIClientSdk::with_config(config);
        let client = OpenAIEmbeddingClient::new(sdk_client, "text-embedding-test".to_string());
        let documents = vec!["doc1_error".to_string()];

        server.mock("POST", "/embeddings")
            .with_status(500) // Simulate server error
            .create_async().await;
        
        let result = client.generate_embeddings(documents).await;
        assert!(result.is_err());
    }


    // --- VoyageAIClient EmbeddingGenerator Trait Tests ---

    #[tokio::test]
    async fn test_voyage_trait_generate_single_embedding_success() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();
        let api_key = "test_voyage_key".to_string();
        let embedding_model = "voyage-test-embed".to_string();
        
        // Instantiate VoyageAIClient with the mock server's URL
        let voyage_client = VoyageAIClient::new(
            test_reqwest_client(),
            api_key.clone(),
            embedding_model.clone(),
            "voyage-rerank-placeholder".to_string(), // Rerank model not used in this test
            Some(mock_url.clone()), // Pass mock server URL
        );
        
        let document = "Voyage test document".to_string();
        let expected_request_payload = VoyageEmbeddingRequest {
            input: VoyageInput::String(document.clone()),
            model: embedding_model.clone(),
            input_type: None, truncation: Some(true), output_dimension: None, output_dtype: None, encoding_format: None,
        };
        let mock_response_body = VoyageEmbeddingResponse {
            object: "list".to_string(),
            data: vec![VoyageEmbeddingData { object: "embedding".to_string(), embedding: vec![0.5, 0.6], index: 0 }],
            model: embedding_model.clone(),
            usage: VoyageUsage { total_tokens: 3 },
        };

        server.mock("POST", "/embeddings") // Mockito will use the base mock_url
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_header("authorization", &format!("Bearer {}", api_key))
            .with_json_body(&expected_request_payload)
            .with_body(serde_json::to_string(&mock_response_body).unwrap())
            .create_async().await;

        // Call the trait method
        let result = voyage_client.generate_single_embedding(document).await;
        
        assert!(result.is_ok(), "Expected Ok, got Err: {:?}", result.err());
        let embedding = result.unwrap();
        assert_eq!(embedding, vec![0.5, 0.6]);
    }

    #[tokio::test]
    async fn test_voyage_trait_generate_single_embedding_error() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();
        let api_key = "test_voyage_key_err".to_string();
        let embedding_model = "voyage-test-embed-err".to_string();

        let voyage_client = VoyageAIClient::new(
            test_reqwest_client(),
            api_key.clone(),
            embedding_model.clone(),
            "voyage-rerank-placeholder".to_string(),
            Some(mock_url.clone()),
        );
        
        let document = "Voyage error document".to_string();
         let expected_request_payload = VoyageEmbeddingRequest { // Request still needs to be matched
            input: VoyageInput::String(document.clone()),
            model: embedding_model.clone(),
            input_type: None, truncation: Some(true), output_dimension: None, output_dtype: None, encoding_format: None,
        };

        server.mock("POST", "/embeddings")
            .with_status(500)
            .with_header("authorization", &format!("Bearer {}", api_key))
            .with_json_body(&expected_request_payload)
            .create_async().await;

        let result = voyage_client.generate_single_embedding(document).await;
        assert!(result.is_err(), "Expected Err, but got Ok");
    }

    #[tokio::test]
    async fn test_voyage_trait_generate_embeddings_success() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();
        let api_key = "test_voyage_multi_key".to_string();
        let embedding_model = "voyage-test-embed-multi".to_string();

        let voyage_client = VoyageAIClient::new(
            test_reqwest_client(),
            api_key.clone(),
            embedding_model.clone(),
            "voyage-rerank-placeholder".to_string(),
            Some(mock_url.clone()),
        );

        let documents = vec!["docA".to_string(), "docB".to_string()];
        let expected_request_payload = VoyageEmbeddingRequest {
            input: VoyageInput::StringArray(documents.clone()),
            model: embedding_model.clone(),
            input_type: None, truncation: Some(true), output_dimension: None, output_dtype: None, encoding_format: None,
        };
        let mock_response_body = VoyageEmbeddingResponse {
            object: "list".to_string(),
            data: vec![
                VoyageEmbeddingData { object: "embedding".to_string(), embedding: vec![0.7], index: 0 },
                VoyageEmbeddingData { object: "embedding".to_string(), embedding: vec![0.8], index: 1 },
            ],
            model: embedding_model.clone(),
            usage: VoyageUsage { total_tokens: 7 },
        };

        server.mock("POST", "/embeddings")
            .with_status(200)
            .with_json_body(&expected_request_payload)
            .with_body(serde_json::to_string(&mock_response_body).unwrap())
            .create_async().await;

        let result = voyage_client.generate_embeddings(documents).await;
        assert!(result.is_ok(), "Expected Ok, got Err: {:?}", result.err());
        let (embeddings, total_tokens) = result.unwrap();
        assert_eq!(embeddings, vec![vec![0.7], vec![0.8]]);
        assert_eq!(total_tokens, 7);
    }

    #[tokio::test]
    async fn test_voyage_trait_generate_embeddings_error() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();
        let api_key = "test_voyage_multi_err_key".to_string();
        let embedding_model = "voyage-test-embed-multi-err".to_string();

        let voyage_client = VoyageAIClient::new(
            test_reqwest_client(),
            api_key.clone(),
            embedding_model.clone(),
            "voyage-rerank-placeholder".to_string(),
            Some(mock_url.clone()),
        );
        
        let documents = vec!["docErr1".to_string(), "docErr2".to_string()];
        let expected_request_payload = VoyageEmbeddingRequest {
            input: VoyageInput::StringArray(documents.clone()),
            model: embedding_model.clone(),
            input_type: None, truncation: Some(true), output_dimension: None, output_dtype: None, encoding_format: None,
        };
        
        server.mock("POST", "/embeddings")
            .with_status(500)
            .with_json_body(&expected_request_payload)
            .create_async().await;

        let result = voyage_client.generate_embeddings(documents).await;
        assert!(result.is_err(), "Expected Err, but got Ok");
    }

    // --- OpenAIReranker Tests ---
    #[tokio::test]
    async fn test_openai_reranker_trait_success() { // Renamed for clarity
        let reranker = OpenAIReranker;
        let query = "test query".to_string();
        let documents = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];
        
        let result = reranker.rerank_documents(query, documents.clone()).await;
        assert!(result.is_ok());
        let reranked_docs = result.unwrap();
        
        assert_eq!(reranked_docs.len(), 3);
        for (i, doc_text) in documents.iter().enumerate() {
            assert_eq!(reranked_docs[i].index, i);
            assert_eq!(reranked_docs[i].text, *doc_text);
            assert_eq!(reranked_docs[i].score, 1.0);
        }
    }

    // --- VoyageAIClient Reranker Tests ---
    // --- VoyageAIClient Reranker Trait Tests ---
    #[tokio::test]
    async fn test_voyage_trait_rerank_documents_success() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();
        let api_key = "test_voyage_rerank_key".to_string();
        let rerank_model = "voyage-test-rerank".to_string();

        let voyage_client = VoyageAIClient::new(
            test_reqwest_client(),
            api_key.clone(),
            "voyage-embed-placeholder".to_string(), // Embedding model not used
            rerank_model.clone(),
            Some(mock_url.clone()), // Pass mock server URL
        );

        let query = "rerank query".to_string();
        let documents_to_rerank = vec!["docX".to_string(), "docY".to_string()];
        let expected_request = VoyageRerankRequest {
            query: query.clone(),
            documents: documents_to_rerank.clone(),
            model: rerank_model.clone(),
            top_k: None, return_documents: Some(true), truncation: Some(true),
        };
        let mock_response_body = VoyageRerankResponse {
            object: "list".to_string(),
            data: vec![
                VoyageRerankData { index: 1, relevance_score: 0.99, document: Some("docY".to_string()) },
                VoyageRerankData { index: 0, relevance_score: 0.88, document: Some("docX".to_string()) },
            ],
            model: rerank_model.clone(),
            usage: VoyageUsage { total_tokens: 15 },
        };
        
        server.mock("POST", "/rerank")
            .with_status(200)
            .with_json_body(&expected_request)
            .with_body(serde_json::to_string(&mock_response_body).unwrap())
            .create_async().await;

        let result = voyage_client.rerank_documents(query, documents_to_rerank).await;
        assert!(result.is_ok(), "Expected Ok, got Err: {:?}", result.err());
        let reranked_docs = result.unwrap();
        assert_eq!(reranked_docs.len(), 2);
        assert_eq!(reranked_docs[0].document, Some("docY".to_string()));
        assert_eq!(reranked_docs[0].relevance_score, 0.99);
    }

    #[tokio::test]
    async fn test_voyage_trait_rerank_documents_error() {
        let mut server = MockServer::new_async().await;
        let mock_url = server.url();
        let api_key = "test_voyage_rerank_err_key".to_string();
        let rerank_model = "voyage-test-rerank-err".to_string();

        let voyage_client = VoyageAIClient::new(
            test_reqwest_client(),
            api_key.clone(),
            "voyage-embed-placeholder".to_string(),
            rerank_model.clone(),
            Some(mock_url.clone()),
        );
        
        let query = "rerank error query".to_string();
        let documents_to_rerank = vec!["docErrX".to_string()];
        let expected_request = VoyageRerankRequest {
            query: query.clone(),
            documents: documents_to_rerank.clone(),
            model: rerank_model.clone(),
            top_k: None, return_documents: Some(true), truncation: Some(true),
        };

        server.mock("POST", "/rerank")
            .with_status(500)
            .with_json_body(&expected_request)
            .create_async().await;

        let result = voyage_client.rerank_documents(query, documents_to_rerank).await;
        assert!(result.is_err(), "Expected Err, but got Ok");
    }
}
