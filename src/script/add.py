import numpy as np
import json
import os

def add_important_edges(filtered_npz_path, merged_json_path, output_npz_path, min_score=3):
    """
    Add important edges based on an already filtered dataset.

    Args:
        filtered_npz_path: Path to the filtered dataset
        merged_json_path: Path to the merged JSON file (contains important edge information)
        output_npz_path: Path to the output final dataset
        min_score: Minimum relevance score threshold (default: 3)
    """
    
    # Read the filtered dataset
    print("Reading filtered dataset...")
    data = np.load(filtered_npz_path, allow_pickle=True)
    
    # Extract data
    node_texts = data['node_texts']
    existing_edges = data['edges']  # shape: (2, num_edges)
    
    # Preserve other fields
    other_data = {}
    for key in data.keys():
        if key not in ['node_texts', 'edges']:
            other_data[key] = data[key]
    
    print("Filtered dataset info:")
    print(f"  - Number of nodes: {len(node_texts)}")
    print(f"  - Number of existing edges: {existing_edges.shape[1]}")
    print(f"  - Other fields: {list(other_data.keys())}")
    
    print("\nReading merged JSON file...")
    with open(merged_json_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    print(f"Merged data count: {len(merged_data)}")
    
    existing_edge_set = set()
    for i in range(existing_edges.shape[1]):
        edge = tuple(sorted([existing_edges[0, i], existing_edges[1, i]]))
        existing_edge_set.add(edge)
    print(f"Existing edge set size: {len(existing_edge_set)}")
    
    print("\nExtracting important edges...")
    important_edges = []
    score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    duplicate_count = 0
    
    for item in merged_data:
        try:
            if isinstance(item['output'], dict):
                relevance_score = item['output'].get('Relevance Score', 0)
            else:
                relevance_score = int(item['output'])

            if relevance_score in score_distribution:
                score_distribution[relevance_score] += 1
            
            if relevance_score >= min_score:
                node_pair = item['node_pair']
                node1, node2 = node_pair[0], node_pair[1]
                
                if 0 <= node1 < len(node_texts) and 0 <= node2 < len(node_texts):
                    edge = tuple(sorted([node1, node2]))
                    if edge not in existing_edge_set:
                        important_edges.append([node1, node2])
                        existing_edge_set.add(edge)
                    else:
                        duplicate_count += 1
                else:
                    print(f"Warning: Invalid node indices {node1}, {node2}")
                    
        except (KeyError, ValueError, TypeError, IndexError) as e:
            print(f"Error processing item: {e}")
            continue
    
    print("\nScore distribution in merged data:")
    for score, count in score_distribution.items():
        print(f"  Score {score}: {count} edges")
    
    print(f"\nImportant edges to add (score >= {min_score}): {len(important_edges)}")
    print(f"Duplicate edges (already exist): {duplicate_count}")
    
    # Merge edges
    print("\nMerging edges...")
    if important_edges:
        important_edges_array = np.array(important_edges).T  # shape: (2, num_new_edges)
        merged_edges = np.concatenate([existing_edges, important_edges_array], axis=1)
    else:
        merged_edges = existing_edges
        print("No new edges to add")
    
    print(f"\nSaving final dataset to: {output_npz_path}")
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)

    save_data = {
        'node_texts': node_texts,
        'edges': merged_edges
    }
    save_data.update(other_data)
    np.savez(output_npz_path, **save_data)
    
    print("Successfully saved final dataset!")
    print(f"  - Original edges: {existing_edges.shape[1]}")
    print(f"  - Added edges: {len(important_edges)}")
    print(f"  - Final edges: {merged_edges.shape[1]}")
    print(f"  - Net change: {merged_edges.shape[1] - existing_edges.shape[1]} edges")
    
    return merged_edges.shape[1], existing_edges.shape[1], len(important_edges)

def validate_final_dataset(npz_path):
    """Validate whether the final dataset is correct."""
    print(f"\nValidating final dataset: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    print("Dataset validation:")
    print(f"  - Node texts shape: {data['node_texts'].shape}")
    print(f"  - Edges shape: {data['edges'].shape}")
    print(f"  - Available fields: {list(data.keys())}")
    
    # Check edge validity
    edges = data['edges']
    num_nodes = len(data['node_texts'])
    
    if edges.shape[0] != 2:
        print("Warning: Edges should have shape (2, num_edges)")
        return False
    
    if np.any(edges < 0) or np.any(edges >= num_nodes):
        print("Warning: Some edge indices are out of bounds")
        return False
    else:
        print("All edge indices are valid")
    
    # Check whether duplicate edges exist
    edge_set = set()
    duplicate_edges = 0
    for i in range(edges.shape[1]):
        edge = tuple(sorted([edges[0, i], edges[1, i]]))
        if edge in edge_set:
            duplicate_edges += 1
        else:
            edge_set.add(edge)
    
    if duplicate_edges > 0:
        print(f"Warning: Found {duplicate_edges} duplicate edges")
    else:
        print("No duplicate edges found")
    
    return True

def get_edge_statistics(npz_path):
    """Get edge statistics."""
    print(f"\nEdge statistics for: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges']
    num_nodes = len(data['node_texts'])
    
    # Compute degree distribution
    degree_count = np.zeros(num_nodes)
    for i in range(edges.shape[1]):
        degree_count[edges[0, i]] += 1
        degree_count[edges[1, i]] += 1
    
    print(f"  - Total edges: {edges.shape[1]}")
    print(f"  - Total nodes: {num_nodes}")
    print(f"  - Average degree: {np.mean(degree_count):.2f}")
    print(f"  - Max degree: {np.max(degree_count)}")
    print(f"  - Min degree: {np.min(degree_count)}")
    print(f"  - Nodes with degree 0: {np.sum(degree_count == 0)}")
    
    return degree_count

# Main function
def main():
    # File path configuration
    filtered_npz = "clean_dataset_new/Amazon_filtered.npz"
    merged_json = "merged_json/Amazon_cosine_similarity_merged_5.json"
    final_npz = "clean_dataset_new/Amazon_final_5.npz"
    
    # Check whether input files exist
    if not os.path.exists(filtered_npz):
        print(f"Error: Filtered dataset not found: {filtered_npz}")
        return
    
    if not os.path.exists(merged_json):
        print(f"Error: Merged JSON file not found: {merged_json}")
        return
    
    # Get statistics of the filtered dataset
    print("Statistics of the filtered dataset:")
    get_edge_statistics(filtered_npz)
    
    # Add important edges
    try:
        final_count, original_count, added_count = add_important_edges(
            filtered_npz_path=filtered_npz,
            merged_json_path=merged_json,
            output_npz_path=final_npz,
            min_score=5
        )
        
        validate_final_dataset(final_npz)
        
        # Get statistics of the final dataset
        print("\nStatistics of the final dataset:")
        get_edge_statistics(final_npz)
        
        print("\nTask completed successfully!")
        print(f"  Filtered dataset: {filtered_npz}")
        print(f"  Merged JSON: {merged_json}")
        print(f"  Final dataset: {final_npz}")
        print(f"  Edge changes: {original_count} → {final_count} (+{added_count})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()