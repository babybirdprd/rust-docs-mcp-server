// Declare modules (keep doc_loader, embeddings, error)
mod doc_loader;
mod embeddings;
mod error;
mod server; // Keep server module as RustDocsServer is defined there
mod services;
mod voyage_client;

// Use necessary items from modules and crates
use crate::{
    doc_loader::Document,
    embeddings::{
        CachedDocumentEmbedding, OpenAIEmbeddingClient, OpenAIReranker, // Added OpenAIReranker
        generate_embeddings_with_paths_and_stats, // Renamed from generate_embeddings
        OPENAI_CLIENT,
    },
    error::ServerError,
    server::RustDocsServer,
    services::{EmbeddingGenerator, Reranker}, // Added Reranker
    voyage_client::VoyageAIClient,
};
use anyhow::Context; // For .context() error handling
use async_openai::{config::OpenAIConfig, Client as OpenAIClient};
use bincode::config;
use reqwest::Client as ReqwestClient; // For VoyageAI
use std::sync::Arc; // For Arc-wrapping clients
use cargo::core::PackageIdSpec;
use clap::Parser; // Import clap Parser
use ndarray::Array1;
// Import rmcp items needed for the new approach
use rmcp::{
    transport::io::stdio, // Use the standard stdio transport
    ServiceExt,           // Import the ServiceExt trait for .serve() and .waiting()
};
use std::{
    collections::hash_map::DefaultHasher,
    env,
    fs::{self, File},
    hash::{Hash, Hasher}, // Import hashing utilities
    io::BufReader,
    path::PathBuf,
};
#[cfg(not(target_os = "windows"))]
use xdg::BaseDirectories;

// --- CLI Argument Parsing ---

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The package ID specification (e.g., "serde@^1.0", "tokio").
    #[arg()] // Positional argument
    package_spec: String,

    /// Optional features to enable for the crate when generating documentation.
    #[arg(short = 'F', long, value_delimiter = ',', num_args = 0..)] // Allow multiple comma-separated values
    features: Option<Vec<String>>,
}

// Helper function to create a stable hash from features
fn hash_features(features: &Option<Vec<String>>) -> String {
    features
        .as_ref()
        .map(|f| {
            let mut sorted_features = f.clone();
            sorted_features.sort_unstable(); // Sort for consistent hashing
            let mut hasher = DefaultHasher::new();
            sorted_features.hash(&mut hasher);
            format!("{:x}", hasher.finish()) // Return hex representation of hash
        })
        .unwrap_or_else(|| "no_features".to_string()) // Use a specific string if no features
}

#[tokio::main]
async fn main() -> Result<(), ServerError> {
    // Load .env file if present
    dotenvy::dotenv().ok();

    // --- Parse CLI Arguments ---
    let cli = Cli::parse();
    let specid_str = cli.package_spec.trim().to_string();
    let features = cli
        .features
        .map(|f| f.into_iter().map(|s| s.trim().to_string()).collect());

    // --- Determine Embedding Provider ---
    let embedding_provider_name = env::var("EMBEDDING_PROVIDER")
        .unwrap_or_else(|_| "openai".to_string())
        .to_lowercase();
    eprintln!("Using embedding provider: {}", embedding_provider_name);


    // Parse the specid string
    let spec = PackageIdSpec::parse(&specid_str).map_err(|e| {
        ServerError::Config(format!(
            "Failed to parse package ID spec '{}': {}",
            specid_str, e
        ))
    })?;

    let crate_name = spec.name().to_string();
    let crate_version_req = spec
        .version()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "*".to_string());

    eprintln!(
        "Target Spec: {}, Parsed Name: {}, Version Req: {}, Features: {:?}",
        specid_str, crate_name, crate_version_req, features
    );

    // --- Determine Paths (incorporating features) ---

    // Sanitize the version requirement string
    let sanitized_version_req = crate_version_req
        .replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");

    // Generate a stable hash for the features to use in the path
    let features_hash = hash_features(&features);

    // Construct the relative path component including provider, features hash
    let cache_filename = format!("embeddings_{}.bin", embedding_provider_name);
    let embeddings_relative_path = PathBuf::from(&crate_name)
        .join(&sanitized_version_req)
        .join(&features_hash) // Add features hash as a directory level
        .join(cache_filename); // Use provider-specific filename

    #[cfg(not(target_os = "windows"))]
    let embeddings_file_path = {
        let xdg_dirs = BaseDirectories::with_prefix("rustdocs-mcp-server")
            .map_err(|e| ServerError::Xdg(format!("Failed to get XDG directories: {}", e)))?;
        xdg_dirs
            .place_data_file(embeddings_relative_path)
            .map_err(ServerError::Io)?
    };

    #[cfg(target_os = "windows")]
    let embeddings_file_path = {
        let cache_dir = dirs::cache_dir().ok_or_else(|| {
            ServerError::Config("Could not determine cache directory on Windows".to_string())
        })?;
        let app_cache_dir = cache_dir.join("rustdocs-mcp-server");
        // Ensure the base app cache directory exists
        fs::create_dir_all(&app_cache_dir).map_err(ServerError::Io)?;
        app_cache_dir.join(embeddings_relative_path)
    };

    eprintln!("Cache file path: {:?}", embeddings_file_path);

    // --- Try Loading Embeddings and Documents from Cache ---
    let mut loaded_from_cache = false;
    let mut loaded_embeddings: Option<Vec<(String, Array1<f32>)>> = None;
    let mut loaded_documents_from_cache: Option<Vec<Document>> = None;

    if embeddings_file_path.exists() {
        eprintln!(
            "Attempting to load cached data from: {:?}",
            embeddings_file_path
        );
        match File::open(&embeddings_file_path) {
            Ok(file) => {
                let reader = BufReader::new(file);
                match bincode::decode_from_reader::<Vec<CachedDocumentEmbedding>, _, _>(
                    reader,
                    config::standard(),
                ) {
                    Ok(cached_data) => {
                        eprintln!(
                            "Successfully loaded {} items from cache. Separating data...",
                            cached_data.len()
                        );
                        let mut embeddings = Vec::with_capacity(cached_data.len());
                        let mut documents = Vec::with_capacity(cached_data.len());
                        for item in cached_data {
                            embeddings.push((item.path.clone(), Array1::from(item.vector)));
                            documents.push(Document {
                                path: item.path,
                                content: item.content,
                            });
                        }
                        loaded_embeddings = Some(embeddings);
                        loaded_documents_from_cache = Some(documents);
                        loaded_from_cache = true;
                    }
                    Err(e) => {
                        eprintln!("Failed to decode cache file: {}. Will regenerate.", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to open cache file: {}. Will regenerate.", e);
            }
        }
    } else {
        eprintln!("Cache file not found. Will generate.");
    }

    // --- Generate or Use Loaded Embeddings ---
    let mut generated_tokens: Option<usize> = None;
    let mut generation_cost: Option<f64> = None; // This might be provider-specific or removed
    let mut documents_for_server: Vec<Document> = loaded_documents_from_cache.unwrap_or_default();

    // --- Initialize OpenAI Client (used by OpenAI embedding service and potentially by server's LLM) ---
    // This client is initialized regardless of the embedding provider,
    // as the server might still use OpenAI for LLM tasks.
    let openai_client_instance = if let Ok(api_base) = env::var("OPENAI_API_BASE") {
        OpenAIClient::with_config(OpenAIConfig::new().with_api_base(api_base))
    } else {
        OpenAIClient::new()
    };
    OPENAI_CLIENT
        .set(openai_client_instance.clone())
        .expect("Failed to set OpenAI client in OnceLock");


    // --- Initialize Embedding Generator Service ---
    let embedding_generator_service: Arc<dyn EmbeddingGenerator> = match embedding_provider_name.as_str() {
        "openai" => {
            let _openai_api_key = env::var("OPENAI_API_KEY").map_err(|_| {
                ServerError::MissingEnvVar("OPENAI_API_KEY (for OpenAI provider)".to_string())
            })?;
            let model_name = env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            Arc::new(OpenAIEmbeddingClient::new(
                openai_client_instance, // Use the client from OnceLock
                model_name,
            ))
        }
        "voyageai" => {
            let api_key = env::var("VOYAGE_API_KEY").map_err(|_| {
                ServerError::MissingEnvVar("VOYAGE_API_KEY (for VoyageAI provider)".to_string())
            })?;
            let model_name = env::var("VOYAGE_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "voyage-large-2-instruct".to_string());
            // Create a new reqwest client for Voyage AI
            let reqwest_client = ReqwestClient::new();
            Arc::new(VoyageAIClient::new(
                reqwest_client,
                api_key,
                model_name,
                env::var("VOYAGE_RERANK_MODEL").unwrap_or_else(|_| "rerank-lite-1".to_string()), // Default rerank model
            ))
        }
        _ => {
            return Err(ServerError::Config(format!(
                "Unsupported embedding provider: {}",
                embedding_provider_name
            )));
        }
    };
    
    // --- Initialize Reranker Service (Placeholder for now) ---
    // This will eventually also depend on a provider choice. For now, default to OpenAIReranker.
    let reranker_service: Arc<dyn Reranker> = Arc::new(OpenAIReranker);


    let final_embeddings = match loaded_embeddings {
        Some(embeddings) => {
            eprintln!("Using embeddings and documents loaded from cache for provider: {}.", embedding_provider_name);
            embeddings
        }
        None => {
            eprintln!("Proceeding with documentation loading and embedding generation using provider: {}.", embedding_provider_name);

            // Load documents if not already loaded (e.g. cache miss for embeddings but docs were loaded)
            if documents_for_server.is_empty() {
                 eprintln!(
                    "Loading documents for crate: {} (Version Req: {}, Features: {:?})",
                    crate_name, crate_version_req, features
                );
                documents_for_server =
                    doc_loader::load_documents(&crate_name, &crate_version_req, features.as_ref())?;
                eprintln!("Loaded {} documents.", documents_for_server.len());
            }


            let (generated_raw_embeddings, total_tokens) = {
                let doc_contents: Vec<String> = documents_for_server
                    .iter()
                    .map(|d| d.content.clone())
                    .collect();
                
                embedding_generator_service
                    .generate_embeddings(doc_contents)
                    .await
                    .map_err(|e| ServerError::Other(format!("Embedding generation failed: {}", e)))?
            };
            
            generated_tokens = Some(total_tokens); // Store total_tokens

            // Calculate cost if OpenAI (or adapt for other providers if they have similar token pricing)
            if embedding_provider_name == "openai" {
                let cost_per_million = 0.02; // Example for text-embedding-3-small
                let estimated_cost = (total_tokens as f64 / 1_000_000.0) * cost_per_million;
                eprintln!(
                    "OpenAI embedding generation cost for {} tokens: ${:.6}",
                    total_tokens, estimated_cost
                );
                generation_cost = Some(estimated_cost);
            } else {
                 eprintln!(
                    "{} embedding generation processed {} tokens.",
                    embedding_provider_name, total_tokens
                );
            }

            // Combine raw embeddings with document paths
            let mut combined_embeddings_with_paths = Vec::new();
            for (i, raw_embedding) in generated_raw_embeddings.into_iter().enumerate() {
                if let Some(doc) = documents_for_server.get(i) {
                    combined_embeddings_with_paths.push((doc.path.clone(), Array1::from(raw_embedding)));
                } else {
                     eprintln!("Warning: Mismatch between document count and embedding count. Skipping embedding for index {}.", i);
                }
            }

            eprintln!(
                "Saving generated documents and embeddings to: {:?}",
                embeddings_file_path
            );

            let mut combined_cache_data: Vec<CachedDocumentEmbedding> = Vec::new();
            let embedding_map: std::collections::HashMap<String, Array1<f32>> =
                combined_embeddings_with_paths.clone().into_iter().collect(); // Use the newly formed combined_embeddings_with_paths

            for doc in &documents_for_server { // Iterate over documents_for_server
                if let Some(embedding_array) = embedding_map.get(&doc.path) {
                    combined_cache_data.push(CachedDocumentEmbedding {
                        path: doc.path.clone(),
                        content: doc.content.clone(),
                        vector: embedding_array.to_vec(),
                    });
                } else {
                    eprintln!(
                        "Warning: Embedding not found for document path: {}. Skipping from cache.",
                        doc.path
                    );
                }
            }
            
            // Cache saving logic (remains largely the same, uses new path)
            match bincode::encode_to_vec(&combined_cache_data, config::standard()) {
                Ok(encoded_bytes) => {
                    if let Some(parent_dir) = embeddings_file_path.parent() {
                        if !parent_dir.exists() {
                            fs::create_dir_all(parent_dir).map_err(|e| {
                                ServerError::Io(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    format!("Failed to create cache directory {}: {}", parent_dir.display(), e),
                                ))
                            })?;
                        }
                    }
                    fs::write(&embeddings_file_path, encoded_bytes).map_err(|e| {
                        ServerError::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to write cache file: {}", e),
                        ))
                    })?;
                    eprintln!("Cache saved successfully ({} items) for provider {}.", combined_cache_data.len(), embedding_provider_name);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to encode data for cache: {}", e);
                }
            }
            combined_embeddings_with_paths // This is the `generated_embeddings`
        }
    };

    // --- Initialize and Start Server ---
    // Note: RustDocsServer::new signature will need to change to accept Arc<dyn EmbeddingGenerator> and Arc<dyn Reranker>
    eprintln!(
        "Initializing server for crate: {} (Version Req: {}, Features: {:?})",
        crate_name, crate_version_req, features
    );

    let features_str = features
        .as_ref()
        .map(|f| format!(" Features: {:?}", f))
        .unwrap_or_default();

    let startup_message = if loaded_from_cache {
        format!(
            "Server for crate '{}' (Version Req: '{}'{}) initialized. Loaded {} embeddings from cache.",
            crate_name, crate_version_req, features_str, final_embeddings.len()
        )
    } else {
        let tokens = generated_tokens.unwrap_or(0);
        let cost = generation_cost.unwrap_or(0.0);
        format!(
            "Server for crate '{}' (Version Req: '{}'{}) initialized. Generated {} embeddings for {} tokens (Est. Cost: ${:.6}).",
            crate_name,
            crate_version_req,
            features_str,
            final_embeddings.len(),
            tokens,
            cost
        )
    };

    // Create the service instance using the updated ::new()
    let service = RustDocsServer::new(
        crate_name.clone(),
        documents_for_server, // These are the full Document structs
        final_embeddings,     // These are Vec<(String path, Array1<f32> embedding)>
        startup_message,
        embedding_generator_service, // Pass the embedding generator
        reranker_service,            // Pass the reranker
    )
    .map_err(|e| ServerError::Other(format!("Failed to create RustDocsServer: {}", e)))?;


    // --- Use standard stdio transport and ServiceExt ---
    eprintln!("Rust Docs MCP server starting via stdio...");

    // Serve the server using the ServiceExt trait and standard stdio transport
    let server_handle = service.serve(stdio()).await.map_err(|e| {
        eprintln!("Failed to start server: {:?}", e);
        ServerError::McpRuntime(e.to_string()) // Use the new McpRuntime variant
    })?;

    eprintln!("{} Docs MCP server running...", &crate_name);

    // Wait for the server to complete (e.g., stdin closed)
    server_handle.waiting().await.map_err(|e| {
        eprintln!("Server encountered an error while running: {:?}", e);
        ServerError::McpRuntime(e.to_string()) // Use the new McpRuntime variant
    })?;

    eprintln!("Rust Docs MCP server stopped.");
    Ok(())
}
