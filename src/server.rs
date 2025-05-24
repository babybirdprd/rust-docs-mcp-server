use crate::{
    doc_loader::Document,
    embeddings::{cosine_similarity, OPENAI_CLIENT}, // OPENAI_CLIENT is still needed for chat
    error::ServerError,
    services::{EmbeddingGenerator, Reranker}, // Import the new traits
};
use anyhow::Context; // For error context
use async_openai::types::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, // CreateEmbeddingRequestArgs is no longer used here directly
                                     // Client as OpenAIClient, // Removed unused import
};
use ndarray::Array1;
use rmcp::model::AnnotateAble; // Import trait for .no_annotation()
use rmcp::{
    Error as McpError,
    Peer,
    ServerHandler, // Import necessary rmcp items
    model::{
        CallToolResult,
        Content,
        GetPromptRequestParam,
        GetPromptResult,
        /* EmptyObject, ErrorCode, */ Implementation,
        ListPromptsResult, // Removed EmptyObject, ErrorCode
        ListResourceTemplatesResult,
        ListResourcesResult,
        LoggingLevel, // Uncommented ListToolsResult
        LoggingMessageNotification,
        LoggingMessageNotificationMethod,
        LoggingMessageNotificationParam,
        Notification,
        PaginatedRequestParam,
        ProtocolVersion,
        RawResource,
        /* Prompt, PromptArgument, PromptMessage, PromptMessageContent, PromptMessageRole, */ // Removed Prompt types
        ReadResourceRequestParam,
        ReadResourceResult,
        Resource,
        ResourceContents,
        ServerCapabilities,
        ServerInfo,
        ServerNotification,
    },
    service::{RequestContext, RoleServer},
    tool,
};
use schemars::JsonSchema; // Import JsonSchema
use serde::Deserialize; // Import Deserialize
use serde_json::json;
use std::{/* borrow::Cow, */ env, sync::Arc}; // Removed borrow::Cow
use tokio::sync::Mutex;

// --- Argument Struct for the Tool ---

#[derive(Debug, Deserialize, JsonSchema)]
struct QueryRustDocsArgs {
    #[schemars(description = "The specific question about the crate's API or usage.")]
    question: String,
    // Removed crate_name field as it's implicit to the server instance
}

// --- Main Server Struct ---

// No longer needs ServerState, holds data directly
#[derive(Clone)] // Add Clone for tool macro requirements
pub struct RustDocsServer {
    crate_name: Arc<String>,
    documents: Arc<Vec<Document>>,
    embeddings: Arc<Vec<(String, Array1<f32>)>>, // Pre-computed embeddings for docs
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>,
    startup_message: Arc<Mutex<Option<String>>>,
    startup_message_sent: Arc<Mutex<bool>>,
    embedding_generator: Arc<dyn EmbeddingGenerator>, // New field
    reranker: Arc<dyn Reranker>,                      // New field
}

impl RustDocsServer {
    // Updated constructor
    pub fn new(
        crate_name: String,
        documents: Vec<Document>,
        embeddings: Vec<(String, Array1<f32>)>, // These are the pre-computed doc embeddings
        startup_message: String,
        embedding_generator: Arc<dyn EmbeddingGenerator>, // New parameter
        reranker: Arc<dyn Reranker>,                      // New parameter
    ) -> Result<Self, ServerError> {
        Ok(Self {
            crate_name: Arc::new(crate_name),
            documents: Arc::new(documents),
            embeddings: Arc::new(embeddings),
            peer: Arc::new(Mutex::new(None)),
            startup_message: Arc::new(Mutex::new(Some(startup_message))),
            startup_message_sent: Arc::new(Mutex::new(false)),
            embedding_generator, // Store the passed service
            reranker,            // Store the passed service
        })
    }

    // Helper function to send log messages via MCP notification
    pub fn send_log(&self, level: LoggingLevel, message: String) {
        let peer_arc = Arc::clone(&self.peer);
        tokio::spawn(async move {
            let mut peer_guard = peer_arc.lock().await;
            if let Some(peer) = peer_guard.as_mut() {
                let params = LoggingMessageNotificationParam {
                    level,
                    logger: None,
                    data: serde_json::Value::String(message),
                };
                let log_notification: LoggingMessageNotification = Notification {
                    method: LoggingMessageNotificationMethod,
                    params,
                };
                let server_notification =
                    ServerNotification::LoggingMessageNotification(log_notification);
                if let Err(e) = peer.send_notification(server_notification).await {
                    eprintln!("Failed to send MCP log notification: {}", e);
                }
            } else {
                eprintln!("Log task ran but MCP peer was not connected.");
            }
        });
    }

    // Helper for creating simple text resources (like in counter example)
    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        doc_loader::Document,
        embeddings::OPENAI_CLIENT as GLOBAL_OPENAI_CLIENT_STATIC, // Alias to avoid confusion
        services::{EmbeddingGenerator, Reranker, VoyageAIClient}, // OpenAIEmbeddingClient not used for Voyage path
    };
    use async_openai::{
        config::OpenAIConfig,
        types::{
            ChatCompletionRequestMessage, ChatCompletionResponseMessage,
            CreateChatCompletionResponse, CreateEmbeddingResponse, Embedding as OpenAIEmbedding,
            Role, Usage as OpenAIUsage,
        },
        Client as OpenAIClientSdk, // Renamed for clarity vs reqwest::Client
    };
    use mockito::{Server as MockServer, Mock};
    use ndarray::Array1;
    use reqwest::Client as ReqwestClient;
    use rmcp::model::ContentValue; // For asserting CallToolResult content
    use serde_json::json;
    use std::sync::Arc;
    use tokio;

    // Helper to create a reqwest client for Voyage tests
    fn test_reqwest_client() -> ReqwestClient {
        ReqwestClient::new()
    }
    
    // Helper to create an OpenAI SDK client pointing to a mock server
    fn mock_openai_sdk_client(mock_server_url: &str) -> OpenAIClientSdk {
        let config = OpenAIConfig::new().with_api_base(mock_server_url);
        OpenAIClientSdk::with_config(config)
    }

    struct TestServerSetup {
        server: RustDocsServer,
        voyage_mock_server: MockServer,
        openai_chat_mock_server: MockServer,
    }

    async fn setup_test_server_for_voyage_scenario(
        crate_name: String,
        docs: Vec<Document>,
        embeddings_map: Vec<(String, Array1<f32>)>, // path -> embedding
    ) -> TestServerSetup {
        let mut voyage_mock_server = MockServer::new_async().await;
        let mut openai_chat_mock_server = MockServer::new_async().await;

        let voyage_base_url = voyage_mock_server.url();
        let openai_chat_base_url = openai_chat_mock_server.url();

        // Setup VoyageAIClient (embedding and reranker)
        let voyage_api_key = "mock_voyage_key".to_string();
        let voyage_embedding_model = "voyage-embed-mock".to_string();
        let voyage_rerank_model = "voyage-rerank-mock".to_string();
        
        let voyage_client_impl = VoyageAIClient::new(
            test_reqwest_client(),
            voyage_api_key,
            voyage_embedding_model,
            voyage_rerank_model,
            Some(voyage_base_url.clone()), // Configure with mock URL
        );
        let embedding_generator: Arc<dyn EmbeddingGenerator> = Arc::new(voyage_client_impl.clone());
        let reranker: Arc<dyn Reranker> = Arc::new(voyage_client_impl);

        // Setup OpenAIClientSdk for chat (to be used by the global static)
        let mock_openai_sdk_for_chat = mock_openai_sdk_client(&openai_chat_base_url);
        
        // Attempt to set the global static. This might fail if already set by another test.
        // Tests using this helper should ideally be run in a way that ensures `set` is called once,
        // or the global static needs to be managed differently for tests (e.g. test-local static).
        // For now, we proceed, acknowledging this potential flakiness in parallel test runs.
        let _ = GLOBAL_OPENAI_CLIENT_STATIC.set(mock_openai_sdk_for_chat);
        if GLOBAL_OPENAI_CLIENT_STATIC.get().is_none() {
            // Fallback if set failed (e.g. in a concurrent test run where another test already set it)
            // This won't point to our mock server, so tests relying on chat mock might fail.
            // This highlights the issue with global statics in tests.
            eprintln!("WARN: GLOBAL_OPENAI_CLIENT_STATIC was already set. Chat mocking might not work as expected for this test.");
        }


        let server_instance = RustDocsServer::new(
            crate_name,
            docs,
            embeddings_map,
            "Test server started".to_string(),
            embedding_generator,
            reranker,
        )
        .expect("Failed to create test server");

        TestServerSetup {
            server: server_instance,
            voyage_mock_server,
            openai_chat_mock_server,
        }
    }

    // --- Test Scenarios ---

    #[tokio::test]
    async fn test_query_rust_docs_voyage_success_path() {
        let docs = vec![
            Document { path: "doc1.md".to_string(), content: "Content of document 1 about Alpha.".to_string() },
            Document { path: "doc2.md".to_string(), content: "Content of document 2 about Beta.".to_string() },
            Document { path: "doc3.md".to_string(), content: "Content of document 3 about Gamma.".to_string() },
        ];
        let embeddings = vec![
            ("doc1.md".to_string(), Array1::from(vec![0.1, 0.2, 0.7])), // Alpha
            ("doc2.md".to_string(), Array1::from(vec![0.3, 0.8, 0.1])), // Beta
            ("doc3.md".to_string(), Array1::from(vec![0.7, 0.1, 0.2])), // Gamma
        ];
        let mut setup = setup_test_server_for_voyage_scenario("testcrate".to_string(), docs.clone(), embeddings).await;

        let question = "Tell me about Beta".to_string();

        // 1. Mock Voyage AI Embedding for the question
        let question_embedding_response = crate::voyage_client::VoyageEmbeddingResponse {
            object: "list".to_string(),
            data: vec![crate::voyage_client::VoyageEmbeddingData {
                object: "embedding".to_string(),
                embedding: vec![0.31, 0.79, 0.11], // Similar to doc2 (Beta)
                index: 0,
            }],
            model: "voyage-embed-mock".to_string(),
            usage: crate::voyage_client::VoyageUsage { total_tokens: 5 },
        };
        let _voyage_embed_mock = setup.voyage_mock_server.mock("POST", "/embeddings")
            .with_status(200)
            .with_body(serde_json::to_string(&question_embedding_response).unwrap())
            .create_async().await;
            // .match_body(Matcher::Json(json!({ // More specific matching if needed
            //     "input": question.clone(), "model": "voyage-embed-mock", "truncation": true 
            // })))

        // 2. Mock Voyage AI Rerank
        // Assume reranker promotes doc2 (Beta) if it wasn't already top, or keeps it top.
        // For this test, doc2 (Beta) content is expected by LLM.
        let rerank_response_data = vec![
            // Simulating that doc2 (Beta) is the top reranked result
            crate::voyage_client::VoyageRerankData { index: 0, relevance_score: 0.95, document: Some(docs[1].content.clone()) }, // Original index of docs[1] within the reranked list (if it was passed based on topN)
            crate::voyage_client::VoyageRerankData { index: 1, relevance_score: 0.90, document: Some(docs[0].content.clone()) },
        ];
        let rerank_response = crate::voyage_client::VoyageRerankResponse {
            object: "list".to_string(), data: rerank_response_data, model: "voyage-rerank-mock".to_string(),
            usage: crate::voyage_client::VoyageUsage { total_tokens: 10 },
        };
         let _voyage_rerank_mock = setup.voyage_mock_server.mock("POST", "/rerank")
            .with_status(200)
            .with_body(serde_json::to_string(&rerank_response).unwrap())
            .create_async().await;

        // 3. Mock OpenAI Chat Completion
        let expected_llm_response_content = "LLM says: Beta is important.".to_string();
        let chat_completion_response = CreateChatCompletionResponse {
            id: "chatcmpl-mockid".to_string(), object: "chat.completion".to_string(), created: 0, model: "gpt-mock".to_string(),
            choices: vec![async_openai::types::ChatChoice {
                index: 0, message: ChatCompletionResponseMessage { role: Role::Assistant, content: Some(expected_llm_response_content.clone()), tool_calls: None, function_call: None }, finish_reason: Some(async_openai::types::FinishReason::Stop), logprobs: None,
            }],
            usage: Some(OpenAIUsage { prompt_tokens: 10, completion_tokens: Some(5), total_tokens: 15 }),
            system_fingerprint: None,
        };
        let _openai_chat_mock = setup.openai_chat_mock_server.mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(serde_json::to_string(&chat_completion_response).unwrap())
            .create_async().await;

        // Call the tool
        let args = QueryRustDocsArgs { question };
        let result = setup.server.query_rust_docs(args).await;

        assert!(result.is_ok(), "query_rust_docs failed: {:?}", result.err());
        let call_result = result.unwrap();
        match call_result.content.get(0).unwrap().value.clone() {
            ContentValue::Text(text_content) => {
                assert!(text_content.text.contains(&expected_llm_response_content));
            }
            _ => panic!("Expected text content from tool result."),
        }
        // Optionally: _voyage_embed_mock.assert_async().await; etc.
    }

    #[tokio::test]
    async fn test_query_rust_docs_voyage_reranker_error_fallback() {
        let docs = vec![
            Document { path: "doc1.md".to_string(), content: "Content of document 1 about Alpha.".to_string() }, // Cosine top
            Document { path: "doc2.md".to_string(), content: "Content of document 2 about Beta.".to_string() },
        ];
        // Question will be most similar to doc1 (Alpha)
        let embeddings = vec![
            ("doc1.md".to_string(), Array1::from(vec![0.9, 0.1, 0.1])), 
            ("doc2.md".to_string(), Array1::from(vec![0.1, 0.9, 0.1])),
        ];
        let mut setup = setup_test_server_for_voyage_scenario("testcrate".to_string(), docs.clone(), embeddings).await;
        
        let question = "Tell me about Alpha".to_string();

        // 1. Mock Voyage AI Embedding for the question (similar to doc1)
        let question_embedding_response = crate::voyage_client::VoyageEmbeddingResponse {
            object: "list".to_string(), data: vec![crate::voyage_client::VoyageEmbeddingData {
                object: "embedding".to_string(), embedding: vec![0.8, 0.12, 0.12], index: 0,
            }], model: "voyage-embed-mock".to_string(), usage: crate::voyage_client::VoyageUsage { total_tokens: 5 },
        };
        let _voyage_embed_mock = setup.voyage_mock_server.mock("POST", "/embeddings")
            .with_status(200).with_body(serde_json::to_string(&question_embedding_response).unwrap()).create_async().await;

        // 2. Mock Voyage AI Rerank to return an error
        let _voyage_rerank_mock = setup.voyage_mock_server.mock("POST", "/rerank")
            .with_status(500).with_body("Reranker failed").create_async().await;

        // 3. Mock OpenAI Chat Completion - should receive content of doc1 (Alpha) due to fallback
        let expected_llm_response_content = "LLM fallback: Alpha is key.".to_string();
        let chat_completion_response = CreateChatCompletionResponse {
            id: "chatcmpl-mockid-fallback".to_string(), object: "chat.completion".to_string(), created: 0, model: "gpt-mock".to_string(),
            choices: vec![async_openai::types::ChatChoice {
                index: 0, message: ChatCompletionResponseMessage { role: Role::Assistant, content: Some(expected_llm_response_content.clone()), tool_calls: None, function_call: None }, finish_reason: Some(async_openai::types::FinishReason::Stop), logprobs: None,
            }],
            usage: Some(OpenAIUsage { prompt_tokens: 10, completion_tokens: Some(5), total_tokens: 15 }), system_fingerprint: None,
        };
        // The mock should expect the content of docs[0] ("... Alpha") in the user prompt.
        let _openai_chat_mock = setup.openai_chat_mock_server.mock("POST", "/chat/completions")
            .with_status(200).with_body(serde_json::to_string(&chat_completion_response).unwrap())
            // .match_body(Matcher::Json(json!({ // This part is tricky to match exactly without seeing the full prompt.
            //     "messages": [
            //         { "role": "system", "content": serde_json::Value::String(...) },
            //         { "role": "user", "content": Matcher::Regex(format!(".*Alpha.*{}", question)) } 
            //     ]
            // })))
            .create_async().await;
            
        let args = QueryRustDocsArgs { question };
        let result = setup.server.query_rust_docs(args).await;

        assert!(result.is_ok(), "query_rust_docs failed on reranker error: {:?}", result.err());
        let call_result = result.unwrap();
        match call_result.content.get(0).unwrap().value.clone() {
            ContentValue::Text(text_content) => {
                assert!(text_content.text.contains(&expected_llm_response_content));
            }
            _ => panic!("Expected text content from tool result."),
        }
    }

    #[tokio::test]
    async fn test_query_rust_docs_voyage_embedding_error() {
        let docs = vec![Document { path: "doc1.md".to_string(), content: "Content".to_string() }];
        let embeddings = vec![("doc1.md".to_string(), Array1::from(vec![0.1, 0.2, 0.3]))];
        let mut setup = setup_test_server_for_voyage_scenario("testcrate".to_string(), docs, embeddings).await;

        let question = "Any question".to_string();

        // 1. Mock Voyage AI Embedding to return an error
        let _voyage_embed_mock = setup.voyage_mock_server.mock("POST", "/embeddings")
            .with_status(500).with_body("Embedding failed").create_async().await;

        // No need to mock rerank or chat as it should fail before.
            
        let args = QueryRustDocsArgs { question };
        let result = setup.server.query_rust_docs(args).await;

        assert!(result.is_err(), "Expected query_rust_docs to fail on embedding error");
        if let Err(e) = result {
            assert!(e.message.contains("Failed to generate question embedding"));
        }
    }
}

// --- Tool Implementation ---

#[tool(tool_box)] // Add tool_box here as well, mirroring the example
// Tool methods go in a regular impl block
impl RustDocsServer {
    // Define the tool using the tool macro
    // Name removed; will be handled dynamically by overriding list_tools/get_tool
    #[tool(
        description = "Query documentation for a specific Rust crate using semantic search and LLM summarization."
    )]
    async fn query_rust_docs(
        &self,
        #[tool(aggr)] // Aggregate arguments into the struct
        args: QueryRustDocsArgs,
    ) -> Result<CallToolResult, McpError> {
        // --- Send Startup Message (if not already sent) ---
        let mut sent_guard = self.startup_message_sent.lock().await;
        if !*sent_guard {
            let mut msg_guard = self.startup_message.lock().await;
            if let Some(message) = msg_guard.take() {
                // Take the message out
                self.send_log(LoggingLevel::Info, message);
                *sent_guard = true; // Mark as sent
            }
            // Drop guards explicitly to avoid holding locks longer than needed
            drop(msg_guard);
            drop(sent_guard);
        } else {
            // Drop guard if already sent
            drop(sent_guard);
        }

        // Argument validation for crate_name removed

        let question = &args.question;

        // Log received query via MCP
        self.send_log(
            LoggingLevel::Info,
            format!(
                "Received query for crate '{}': {}",
                self.crate_name, question
            ),
        );

        // --- Embedding Generation for Question using the service ---
        let question_embedding_vec = self
            .embedding_generator
            .generate_single_embedding(question.clone()) // Pass the question string
            .await
            .map_err(|e| {
                McpError::internal_error(
                    format!("Failed to generate question embedding: {}", e),
                    None,
                )
            })?;
        
        let question_vector = Array1::from(question_embedding_vec);

        // --- Initial Candidate Selection (Top N by Cosine Similarity) ---
        let mut all_doc_scores: Vec<(f32, String)> = self
            .embeddings
            .iter()
            .map(|(path, doc_embedding)| {
                let score = cosine_similarity(question_vector.view(), doc_embedding.view());
                (score, path.clone())
            })
            .collect();

        // Sort by score in descending order
        all_doc_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        const TOP_N_CANDIDATES: usize = 5;
        let top_candidates_paths: Vec<String> = all_doc_scores
            .into_iter()
            .take(TOP_N_CANDIDATES)
            .map(|(_score, path)| path)
            .collect();

        let final_context_document_content: Option<String>;

        if top_candidates_paths.is_empty() {
            self.send_log(LoggingLevel::Info, "No document candidates found after initial search.".to_string());
            final_context_document_content = None;
        } else {
            // Retrieve content for top N candidates
            let candidate_docs_for_reranking: Vec<String> = top_candidates_paths
                .iter()
                .filter_map(|path| {
                    self.documents
                        .iter()
                        .find(|doc| doc.path == *path)
                        .map(|doc| doc.content.clone())
                })
                .collect();

            if candidate_docs_for_reranking.is_empty() {
                 self.send_log(LoggingLevel::Warn, "Could not retrieve content for any top candidate paths.".to_string());
                 final_context_document_content = None; // Or fall back to top_candidates_paths[0] if content was found for that
            } else {
                // --- Perform Reranking ---
                self.send_log(
                    LoggingLevel::Info,
                    format!("Performing reranking for {} candidates.", candidate_docs_for_reranking.len()),
                );

                match self
                    .reranker
                    .rerank_documents(question.clone(), candidate_docs_for_reranking.clone()) // Pass contents
                    .await
                {
                    Ok(reranked_docs) => {
                        if let Some(top_reranked_doc) = reranked_docs.first() {
                            // The reranked_docs text field should contain the document content
                            // as per VoyageAIClient implementation.
                            final_context_document_content = Some(top_reranked_doc.text.clone());
                            self.send_log(
                                LoggingLevel::Info,
                                format!(
                                    "Top document after reranking (original index {}): Score {:.4}",
                                    top_reranked_doc.index, top_reranked_doc.score
                                ),
                            );
                        } else {
                            self.send_log(
                                LoggingLevel::Warn,
                                "Reranker returned no documents. Falling back to top cosine similarity document.".to_string(),
                            );
                            // Fallback: use the content of the best document from cosine similarity
                            final_context_document_content = candidate_docs_for_reranking.first().cloned();
                        }
                    }
                    Err(e) => {
                        self.send_log(
                            LoggingLevel::Error,
                            format!("Reranking failed: {}. Falling back to top cosine similarity document.", e),
                        );
                        // Fallback: use the content of the best document from cosine similarity
                        final_context_document_content = candidate_docs_for_reranking.first().cloned();
                    }
                }
            }
        }


        // --- Generate Response using LLM ---
        let response_text = match final_context_document_content {
            Some(context_content) => {
                // LLM call for summarization - uses OPENAI_CLIENT (static)
                let client = OPENAI_CLIENT.get().ok_or_else(|| {
                        McpError::internal_error("OpenAI client for chat not initialized", None)
                    })?;

                    let system_prompt = format!(
                        "You are an expert technical assistant for the Rust crate '{}'. \
                         Answer the user's question based *only* on the provided context. \
                         If the context does not contain the answer, say so. \
                         Do not make up information. Be clear, concise, and comprehensive providing example usage code when possible.",
                        self.crate_name
                    );
                    let user_prompt = format!(
                        "Context:\n---\n{}\n---\n\nQuestion: {}",
                        context_content, question // Use context_content here
                    );

                    let llm_model: String = env::var("LLM_MODEL")
                        .unwrap_or_else(|_| "gpt-4o-mini-2024-07-18".to_string());
                    let chat_request = CreateChatCompletionRequestArgs::default()
                        .model(llm_model)
                        .messages(vec![
                            ChatCompletionRequestSystemMessageArgs::default()
                                .content(system_prompt)
                                .build()
                                .map_err(|e| McpError::internal_error(format!("Failed to build system message: {}", e), None))?
                                .into(),
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(user_prompt)
                                .build()
                                .map_err(|e| McpError::internal_error(format!("Failed to build user message: {}", e), None))?
                                .into(),
                        ])
                        .build()
                        .map_err(|e| McpError::internal_error(format!("Failed to build chat request: {}", e), None))?;

                    let chat_response = client
                        .chat()
                        .create(chat_request)
                        .await
                        .map_err(|e| McpError::internal_error(format!("OpenAI chat API error: {}", e), None))?;

                    chat_response
                        .choices
                        .first()
                        .and_then(|choice| choice.message.content.clone())
                        .unwrap_or_else(|| "Error: No response from LLM.".to_string())
                } else {
                    "Error: Could not find content for best matching document.".to_string()
                }
            }
            None => "Could not find any relevant document context.".to_string(),
        };

        // --- Format and Return Result ---
        Ok(CallToolResult::success(vec![Content::text(format!(
            "From {} docs: {}",
            self.crate_name, response_text
        ))]))
    }
}

// --- ServerHandler Implementation ---

#[tool(tool_box)] // Use imported tool macro directly
impl ServerHandler for RustDocsServer {
    fn get_info(&self) -> ServerInfo {
        // Define capabilities using the builder
        let capabilities = ServerCapabilities::builder()
            .enable_tools() // Enable tools capability
            .enable_logging() // Enable logging capability
            // Add other capabilities like resources, prompts if needed later
            .build();

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05, // Use latest known version
            capabilities,
            server_info: Implementation {
                name: "rust-docs-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            // Provide instructions based on the specific crate
            instructions: Some(format!(
                "This server provides tools to query documentation for the '{}' crate. \
                 Use the 'query_rust_docs' tool with a specific question to get information \
                 about its API, usage, and examples, derived from its official documentation.",
                self.crate_name
            )),
        }
    }

    // --- Placeholder Implementations for other ServerHandler methods ---
    // Implement these properly if resource/prompt features are added later.

    async fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        // Example: Return the crate name as a resource
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text(&format!("crate://{}", self.crate_name), "crate_name"),
            ],
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let expected_uri = format!("crate://{}", self.crate_name);
        if request.uri == expected_uri {
            Ok(ReadResourceResult {
                contents: vec![ResourceContents::text(
                    self.crate_name.as_str(), // Explicitly get &str from Arc<String>
                    &request.uri,
                )],
            })
        } else {
            Err(McpError::resource_not_found(
                format!("Resource URI not found: {}", request.uri),
                Some(json!({ "uri": request.uri })),
            ))
        }
    }

    async fn list_prompts(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult {
            next_cursor: None,
            prompts: Vec::new(), // No prompts defined yet
        })
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        Err(McpError::invalid_params(
            // Or prompt_not_found if that exists
            format!("Prompt not found: {}", request.name),
            None,
        ))
    }

    async fn list_resource_templates(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(), // No templates defined yet
        })
    }
}
