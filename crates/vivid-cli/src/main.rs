//! # Vivid CLI
//!
//! Command-line utility to interact with the Vivid vector search engine.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vivid_index::HnswIndex;

#[derive(Parser)]
#[command(name = "vivid")]
#[command(about = "Vivid - High-performance Vector Search Engine CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new (empty) index file
    Create {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Dimension of the vectors
        #[arg(short = 'd', long)]
        dimension: usize,

        /// Overwrite if the file already exists
        #[arg(short = 'f', long)]
        force: bool,
    },
    /// Insert a new vector (fails if ID already exists)
    Insert {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Unique identifier for the vector
        #[arg(short = 'n', long)]
        id: u64,

        /// Vector components as a JSON array (e.g., "[0.1, 0.2, 0.3]")
        #[arg(short = 'v', long)]
        vector: String,
    },
    /// Update an existing vector by ID (fails if ID not found)
    Update {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Identifier of the vector to update
        #[arg(short = 'n', long)]
        id: u64,

        /// New vector components as a JSON array
        #[arg(short = 'v', long)]
        vector: String,
    },
    /// Delete a vector by ID
    Delete {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Identifier of the vector to delete
        #[arg(short = 'n', long)]
        id: u64,
    },
    /// Lookup a vector by ID and print it
    Get {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Identifier to look up
        #[arg(short = 'n', long)]
        id: u64,
    },
    /// Show index statistics
    Info {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,
    },
    /// Insert or update a vector (upsert semantics)
    Upsert {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Dimension of the vector (only used when creating a new index)
        #[arg(short = 'd', long, default_value_t = 3)]
        dimension: usize,

        /// Unique identifier for the vector
        #[arg(short = 'n', long)]
        id: u64,

        /// Vector components as a JSON array
        #[arg(short = 'v', long)]
        vector: String,
    },
    /// Search for the nearest neighbors of a query vector
    Search {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Query vector as a JSON array
        #[arg(short = 'q', long)]
        query: String,

        /// Number of top results to return
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
    },
}

fn load_index(index: &PathBuf) -> Result<HnswIndex, Box<dyn std::error::Error>> {
    if !index.exists() {
        eprintln!("Error: index file {} does not exist.", index.display());
        std::process::exit(1);
    }
    Ok(HnswIndex::load_from_file(index)?)
}

fn parse_vector(s: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    serde_json::from_str(s)
        .map_err(|_| "Invalid JSON vector format. Expected an array of numbers like '[1.0, 2.0]'".into())
}

fn save_and_report(index: &HnswIndex, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    index.save_to_file(path)?;
    println!("Saved index to {} ({} vectors).", path.display(), index.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Create { index, dimension, force } => {
            if index.exists() && !force {
                eprintln!("Error: file {} already exists. Use --force to overwrite.", index.display());
                std::process::exit(1);
            }
            let hnsw = HnswIndex::new(dimension);
            hnsw.save_to_file(&index)?;
            println!("Created empty index at {} (dim={}).", index.display(), dimension);
        }
        Commands::Insert { index, id, vector } => {
            let mut hnsw = load_index(&index)?;
            let vec_data = parse_vector(&vector)?;
            hnsw.insert(id, vec_data)?;
            save_and_report(&hnsw, &index)?;
        }
        Commands::Update { index, id, vector } => {
            let mut hnsw = load_index(&index)?;
            let vec_data = parse_vector(&vector)?;
            hnsw.update(id, vec_data)?;
            save_and_report(&hnsw, &index)?;
        }
        Commands::Delete { index, id } => {
            let mut hnsw = load_index(&index)?;
            hnsw.remove(id)?;
            save_and_report(&hnsw, &index)?;
        }
        Commands::Get { index, id } => {
            let hnsw = load_index(&index)?;
            match hnsw.get(id) {
                Some(vec) => println!("ID {}: {:?}", id, vec),
                None => eprintln!("ID {} not found.", id),
            }
        }
        Commands::Info { index } => {
            let hnsw = load_index(&index)?;
            println!("Index: {}", index.display());
            println!("  Dimension: {}", hnsw.dimension());
            println!("  Vectors:   {}", hnsw.len());
        }
        Commands::Upsert { index, dimension, id, vector } => {
            let vec_data = parse_vector(&vector)?;
            let mut hnsw = if index.exists() {
                HnswIndex::load_from_file(&index)?
            } else {
                println!("Creating new index with dimension {}...", dimension);
                HnswIndex::new(dimension)
            };

            if hnsw.contains(id) {
                hnsw.update(id, vec_data)?;
                println!("Updated vector ID {}.", id);
            } else {
                hnsw.insert(id, vec_data)?;
                println!("Inserted vector ID {}.", id);
            }
            save_and_report(&hnsw, &index)?;
        }
        Commands::Search { index, query, top_k } => {
            let hnsw = load_index(&index)?;
            let vec_query = parse_vector(&query)?;

            println!("Searching top {} nearest vectors...", top_k);
            let start_time = std::time::Instant::now();
            let hits = hnsw.search(&vec_query, top_k)?;
            let duration = start_time.elapsed();

            println!("\nSearch completed in {:?}", duration);
            println!("--------------------------------------");
            for (i, hit) in hits.iter().enumerate() {
                println!("{}. ID: {} | Distance: {:.6}", i + 1, hit.id, hit.score);
            }
        }
    }

    Ok(())
}
