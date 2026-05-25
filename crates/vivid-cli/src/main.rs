//! # Vivid CLI
//!
//! Command-line utility to interact with the Vivid vector search engine.
//!
//! Index type is auto-detected from file magic bytes:
//! - `VIDH` → HnswIndex (supports all operations)
//! - `VIDV` → FlatIndex (all operations; O(n) for insert/delete due to linear scan)

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vivid_index::{FlatIndex, HnswIndex};

#[derive(Parser)]
#[command(name = "vivid")]
#[command(about = "Vivid - High-performance Vector Search Engine CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new index (type: hnsw | flat; default: hnsw)
    Create {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Dimension of the vectors
        #[arg(short = 'd', long)]
        dimension: usize,

        /// Index type: hnsw (default) or flat
        #[arg(short = 't', long, default_value = "hnsw", value_parser = ["hnsw", "flat"])]
        index_type: String,

        /// Overwrite if the file already exists
        #[arg(short = 'f', long)]
        force: bool,
    },
    /// Insert a new vector into an HNSW index (fails if ID already exists)
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
    /// Update an existing vector in an HNSW index (fails if ID not found)
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
    /// Delete a vector from an HNSW index by ID
    Delete {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Identifier of the vector to delete
        #[arg(short = 'n', long)]
        id: u64,
    },
    /// Lookup a vector by ID (works with both HNSW and Flat indexes)
    Get {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Identifier to look up
        #[arg(short = 'n', long)]
        id: u64,
    },
    /// Show index statistics (works with both HNSW and Flat indexes)
    Info {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,
    },
    /// Insert or update a vector in an HNSW index
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
    /// Insert multiple vectors from a JSON file (fails if any ID already exists)
    BatchInsert {
        /// Path to the binary index file
        #[arg(short = 'i', long)]
        index: PathBuf,

        /// Path to a JSON file containing an array of [id, vector] pairs
        /// (e.g., [[1, [0.1, 0.2]], [2, [0.3, 0.4]]])
        #[arg(short = 'f', long)]
        file: PathBuf,
    },
    /// Search for the nearest neighbors (works with both HNSW and Flat indexes)
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

/// Creates an empty FlatIndex file (VIDV header with 0 vectors).
fn create_empty_flat(path: &PathBuf, dimension: usize) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut f = std::fs::File::create(path.as_path())?;
    f.write_all(b"VIDV")?;
    f.write_all(&(dimension as u32).to_le_bytes())?;
    f.write_all(&0u64.to_le_bytes())?;
    f.flush()?;
    Ok(())
}

/// Reads the magic bytes of a file to determine its index type.
fn detect_index_type(path: &PathBuf) -> Result<&'static str, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|_| format!("Cannot open file {}", path.display()))?;
    let mut magic = [0u8; 4];
    use std::io::Read;
    file.read_exact(&mut magic)
        .map_err(|_| format!("Cannot read magic bytes from {}", path.display()))?;
    match &magic {
        b"VIDH" => Ok("hnsw"),
        b"VIDV" => Ok("flat"),
        _ => Err(format!("Unknown index format (magic: {:?})", magic)),
    }
}

fn require_index(path: &PathBuf) -> PathBuf {
    if !path.exists() {
        eprintln!("Error: file {} does not exist.", path.display());
        std::process::exit(1);
    }
    path.clone()
}

fn parse_vector(s: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    serde_json::from_str(s)
        .map_err(|_| "Invalid JSON vector format. Expected an array of numbers like '[1.0, 2.0]'".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        // ── create (hnsw, default | flat) ──
        Commands::Create { index, dimension, index_type, force } => {
            if index.exists() && !force {
                eprintln!("Error: file {} already exists. Use --force to overwrite.", index.display());
                std::process::exit(1);
            }
            match index_type.as_str() {
                "hnsw" => {
                    let hnsw = HnswIndex::new(dimension);
                    hnsw.save_to_file(&index)?;
                    println!("Created empty HNSW index at {} (dim={}).", index.display(), dimension);
                }
                "flat" => {
                    create_empty_flat(&index, dimension)?;
                    println!("Created empty Flat index at {} (dim={}).", index.display(), dimension);
                }
                _ => unreachable!(),
            }
        }

        // ── insert (HNSW or Flat) ──
        Commands::Insert { index, id, vector } => {
            let path = require_index(&index);
            let vec_data = parse_vector(&vector)?;
            let count = match detect_index_type(&path)? {
                "hnsw" => {
                    let mut hnsw = HnswIndex::load_from_file(&path)?;
                    hnsw.insert(id, vec_data)?;
                    let count = hnsw.len();
                    hnsw.save_to_file(&path)?;
                    count
                }
                "flat" => {
                    let mut flat = FlatIndex::open(&path)?;
                    flat.insert(id, vec_data)?;
                    let count = flat.len();
                    flat.save_to_file(&path)?;
                    count
                }
                _ => unreachable!(),
            };
            println!("Inserted ID {} ({} vectors total).", id, count);
        }

        // ── update (HNSW or Flat) ──
        Commands::Update { index, id, vector } => {
            let path = require_index(&index);
            let vec_data = parse_vector(&vector)?;
            let count = match detect_index_type(&path)? {
                "hnsw" => {
                    let mut hnsw = HnswIndex::load_from_file(&path)?;
                    hnsw.update(id, vec_data)?;
                    let count = hnsw.len();
                    hnsw.save_to_file(&path)?;
                    count
                }
                "flat" => {
                    let mut flat = FlatIndex::open(&path)?;
                    flat.update(id, vec_data)?;
                    let count = flat.len();
                    flat.save_to_file(&path)?;
                    count
                }
                _ => unreachable!(),
            };
            println!("Updated ID {} ({} vectors total).", id, count);
        }

        // ── delete (HNSW or Flat) ──
        Commands::Delete { index, id } => {
            let path = require_index(&index);
            match detect_index_type(&path)? {
                "hnsw" => {
                    let mut hnsw = HnswIndex::load_from_file(&path)?;
                    hnsw.remove(id)?;
                    hnsw.save_to_file(&path)?;
                }
                "flat" => {
                    let mut flat = FlatIndex::open(&path)?;
                    flat.remove(id)?;
                    flat.save_to_file(&path)?;
                }
                _ => unreachable!(),
            }
            println!("Deleted ID {}.", id);
        }

        // ── get (both HNSW and Flat) ──
        Commands::Get { index, id } => {
            let path = require_index(&index);
            match detect_index_type(&path)? {
                "hnsw" => {
                    let hnsw = HnswIndex::load_from_file(&path)?;
                    match hnsw.get(id) {
                        Some(vec) => println!("ID {}: {:?}", id, vec),
                        None => eprintln!("ID {} not found.", id),
                    }
                }
                "flat" => {
                    let flat = FlatIndex::open(&path)?;
                    match flat.get(id) {
                        Some(vec) => println!("ID {}: {:?}", id, vec),
                        None => eprintln!("ID {} not found.", id),
                    }
                }
                _ => unreachable!(),
            }
        }

        // ── info (both HNSW and Flat) ──
        Commands::Info { index } => {
            let path = require_index(&index);
            match detect_index_type(&path)? {
                "hnsw" => {
                    let hnsw = HnswIndex::load_from_file(&path)?;
                    println!("Index: {} (HNSW)", path.display());
                    println!("  Dimension: {}", hnsw.dimension());
                    println!("  Vectors:   {}", hnsw.len());
                }
                "flat" => {
                    let flat = FlatIndex::open(&path)?;
                    println!("Index: {} (Flat, exact)", path.display());
                    println!("  Dimension: {}", flat.dimension());
                    println!("  Vectors:   {}", flat.len());
                }
                _ => unreachable!(),
            }
        }

        // ── upsert (HNSW or Flat) ──
        Commands::Upsert { index, dimension, id, vector } => {
            let vec_data = parse_vector(&vector)?;
            if index.exists() {
                let path = require_index(&index);
                match detect_index_type(&path)? {
                    "hnsw" => {
                        let mut hnsw = HnswIndex::load_from_file(&path)?;
                        if hnsw.contains(id) {
                            hnsw.update(id, vec_data)?;
                            println!("Updated ID {}.", id);
                        } else {
                            hnsw.insert(id, vec_data)?;
                            println!("Inserted ID {}.", id);
                        }
                        hnsw.save_to_file(&path)?;
                    }
                    "flat" => {
                        let mut flat = FlatIndex::open(&path)?;
                        if flat.contains(id) {
                            flat.update(id, vec_data)?;
                            println!("Updated ID {}.", id);
                        } else {
                            flat.insert(id, vec_data)?;
                            println!("Inserted ID {}.", id);
                        }
                        flat.save_to_file(&path)?;
                    }
                    _ => unreachable!(),
                }
            } else {
                println!("Creating new HNSW index with dimension {}...", dimension);
                let mut hnsw = HnswIndex::new(dimension);
                hnsw.insert(id, vec_data)?;
                hnsw.save_to_file(&index)?;
                println!("Inserted ID {}.", id);
            }
            println!("Saved.");
        }

        // ── batch-insert (HNSW or Flat) ──
        Commands::BatchInsert { index, file } => {
            let path = require_index(&index);
            let data = std::fs::read_to_string(&file)?;
            let items: Vec<(u64, Vec<f32>)> = serde_json::from_str(&data)
                .map_err(|e| format!("Invalid JSON file {}: {}", file.display(), e))?;

            let count = match detect_index_type(&path)? {
                "hnsw" => {
                    let mut hnsw = HnswIndex::load_from_file(&path)?;
                    hnsw.insert_batch(&items)?;
                    let count = hnsw.len();
                    hnsw.save_to_file(&path)?;
                    count
                }
                "flat" => {
                    let mut flat = FlatIndex::open(&path)?;
                    for (id, vec) in &items {
                        flat.insert(*id, vec.clone())?;
                    }
                    let count = flat.len();
                    flat.save_to_file(&path)?;
                    count
                }
                _ => unreachable!(),
            };
            println!("Inserted {} vectors ({} total).", items.len(), count);
        }

        // ── search (both HNSW and Flat) ──
        Commands::Search { index, query, top_k } => {
            let path = require_index(&index);
            let vec_query = parse_vector(&query)?;

            let start_time = std::time::Instant::now();
            let (hits, index_type) = match detect_index_type(&path)? {
                "hnsw" => {
                    let hnsw = HnswIndex::load_from_file(&path)?;
                    (hnsw.search(&vec_query, top_k)?, "HNSW")
                }
                "flat" => {
                    let flat = FlatIndex::open(&path)?;
                    (flat.search(&vec_query, top_k)?, "Flat (exact)")
                }
                _ => unreachable!(),
            };
            let duration = start_time.elapsed();

            println!("Search top {} via {}: completed in {:?}", top_k, index_type, duration);
            println!("--------------------------------------");
            for (i, hit) in hits.iter().enumerate() {
                println!("{}. ID: {} | Distance: {:.6}", i + 1, hit.id, hit.score);
            }
        }
    }

    Ok(())
}
