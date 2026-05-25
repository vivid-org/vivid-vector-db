//! # Vivid CLI
//!
//! Command-line utility to interact with the Vivid vector database engine.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vivid_index::FlatIndex;

#[derive(Parser)]
#[command(name = "vivid")]
#[command(about = "Vivid - High-performance Vector Search Engine CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inserts or updates a vector in the specified index file
    Upsert {
        /// Path to the binary index file
        #[arg(short, long)]
        index: PathBuf,

        /// Dimension of the vector (only required if creating a new index)
        #[arg(short, long, default_value_t = 3)]
        dimension: usize,

        /// Unique identifier for the vector
        #[arg(short, long)]
        id: u64,

        /// Vector components as a JSON array string (e.g., "[0.1, 0.2, 0.3]")
        #[arg(short, long)]
        vector: String,
    },
    /// Searches for the nearest neighbors of a query vector
    Search {
        /// Path to the binary index file
        #[arg(short, long)]
        index: PathBuf,

        /// Query vector as a JSON array string (e.g., "[0.1, 0.2, 0.3]")
        #[arg(short, long)]
        query: String,

        /// Number of top closest results to return
        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Upsert { index, dimension, id, vector } => {
            let vec_data: Vec<f32> = serde_json::from_str(&vector)
                .map_err(|_| "Invalid JSON vector format. Expected an array of numbers like '[1.0, 2.0]'")?;

            let mut flat_index = if index.exists() {
                println!("Loading existing index from {}...", index.display());
                FlatIndex::load_from_file(&index)?
            } else {
                println!("Creating a new index with dimension {}...", dimension);
                FlatIndex::new(dimension)
            };

            flat_index.insert(id, vec_data)?;
            
            let index_path_clone = index.clone();
            tokio::task::spawn_blocking(move || {
                flat_index.save_to_file(index_path_clone)
            }).await??;

            println!("Successfully upserted vector ID {} and saved index to Disk.", id);
        }
        Commands::Search { index, query, top_k } => {
            if !index.exists() {
                eprintln!("Error: Index file {} does not exist.", index.display());
                std::process::exit(1);
            }

            let vec_query: Vec<f32> = serde_json::from_str(&query)
                .map_err(|_| "Invalid JSON query format.")?;

            let flat_index = FlatIndex::load_from_file(&index)?;
            
            println!("Searching top {} nearest vectors...", top_k);
            let start_time = std::time::Instant::now();
            
            let hits = flat_index.search(&vec_query, top_k)?;
            let duration = start_time.elapsed();

            println!("\nSearch completed in {:?}", duration);
            println!("--------------------------------------");
            for (i, hit) in hits.iter().enumerate() {
                println!("{}. ID: {} | Distance Score: {:.6}", i + 1, hit.id, hit.score);
            }
        }
    }

    Ok(())
}
