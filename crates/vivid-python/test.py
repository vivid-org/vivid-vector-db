import vivid

# 1. Initialize our Rust vector index via Python
index = vivid.PyFlatIndex(3)

# 2. Insert some 3D vectors
index.insert(101, [1.0, 0.0, 0.0]) # X axis
index.insert(102, [0.0, 1.0, 0.0]) # Y axis
index.insert(103, [0.0, 0.0, 1.0]) # Z axis

# 3. Check the total number of items
print(f"Total vectors in index: {len(index)}")

# 4. Perform a vector search near the Y axis
# Our nightly manual SIMD will handle this under the hood!
query_vector = [0.1, 0.9, 0.0]
top_hits = index.search(query_vector, top_k=2)

print("\nSearch results from Rust:")
for i, hit in enumerate(top_hits):
    print(f"Match {i+1}: ID={hit['id']}, Distance={hit['score']:.6f}")

