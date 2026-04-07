import numpy as np
import json
import os

def filter_edges_by_relevance(npz_path, output_json_path, save_path, min_score=3):
    """
    Filter edges based on LLM relevance scores, removing edges with low relevance
    
    Args:
        npz_path: Path to the original dataset
        output_json_path: Path to the LLM output results
        save_path: Path to save the filtered dataset
        min_score: Minimum relevance score threshold
    """
    
    # Read original dataset
    print("Reading original dataset...")
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract original data
    node_texts = data['node_texts']
    edges = data['edges']  # shape: (2, num_edges)
    
    # Preserve other fields if they exist
    other_data = {}
    for key in data.keys():
        if key not in ['node_texts', 'edges']:
            other_data[key] = data[key]
    
    print(f"Original dataset info:")
    print(f"  - Number of nodes: {len(node_texts)}")
    print(f"  - Number of edges: {edges.shape[1]}")
    print(f"  - Other fields: {list(other_data.keys())}")
    
    # Read LLM output results
    print("\nReading LLM output results...")
    with open(output_json_path, 'r', encoding='utf-8') as f:
        llm_results = json.load(f)
    
    print(f"LLM results count: {len(llm_results)}")
    
    # Ensure LLM results match edge count
    if len(llm_results) != edges.shape[1]:
        print(f"Warning: LLM results count ({len(llm_results)}) doesn't match edge count ({edges.shape[1]})")
        print("Using minimum count for processing...")
        max_count = min(len(llm_results), edges.shape[1])
    else:
        max_count = len(llm_results)
    
    # Extract relevance scores
    print("\nExtracting relevance scores...")
    valid_edges = []
    score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for i in range(max_count):
        try:
            if isinstance(llm_results[i]['output'], dict):
                relevance_score = llm_results[i]['output'].get('Relevance Score', 0)
            else:
                relevance_score = int(llm_results[i]['output'])

            if relevance_score in score_distribution:
                score_distribution[relevance_score] += 1
            
            if relevance_score >= min_score:
                valid_edges.append(i)
                
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing result {i}: {e}")
            continue
    
    print(f"\n Score distribution:")
    for score, count in score_distribution.items():
        print(f"  Score {score}: {count} edges")
    print(f"\n Valid edges (score >= {min_score}): {len(valid_edges)}")
    print(f"Filtered out edges: {edges.shape[1] - len(valid_edges)}")
    
    # Build filtered edge set
    print("\n Building filtered edge set...")
    filtered_edges = edges[:, valid_edges]
    print(f"\n Saving filtered dataset to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_data = {
        'node_texts': node_texts,
        'edges': filtered_edges
    }
    
    save_data.update(other_data)
    np.savez(save_path, **save_data)
    print(f" Successfully saved filtered dataset!")
    print(f"   - Original edges: {edges.shape[1]}")
    print(f"   - Filtered edges: {filtered_edges.shape[1]}")
    print(f"   - Reduction: {((edges.shape[1] - filtered_edges.shape[1]) / edges.shape[1] * 100):.1f}%")
    return filtered_edges.shape[1], edges.shape[1]

def validate_filtered_dataset(npz_path):
    print(f"\nValidating filtered dataset: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"Dataset validation:")
    print(f"  - Node texts shape: {data['node_texts'].shape}")
    print(f"  - Edges shape: {data['edges'].shape}")
    print(f"  - Available fields: {list(data.keys())}")
    
    edges = data['edges']
    num_nodes = len(data['node_texts'])
    
    if edges.shape[0] != 2:
        print(" Warning: Edges should have shape (2, num_edges)")
    
    if np.any(edges < 0) or np.any(edges >= num_nodes):
        print(" Warning: Some edge indices are out of bounds")
    else:
        print("All edge indices are valid")
    
    return True

def main():
    original_npz = "../../dataset/Cornell.npz"
    llm_output_json = "../data/output_json/Cornell_edgeset_output_cleaned.json"
    filtered_npz = "../../dataset/Cornell_filtered.npz"
    
    # Execute edge filtering
    try:
        filtered_count, original_count = filter_edges_by_relevance(
            npz_path=original_npz,
            output_json_path=llm_output_json,
            save_path=filtered_npz,
            min_score=5 
        )
        
        # Validate filtered dataset
        validate_filtered_dataset(filtered_npz)
        
        print(f"\nTask completed successfully!")
        print(f"   Original dataset: {original_npz}")
        print(f"   Filtered dataset: {filtered_npz}")
        print(f"   Edge reduction: {original_count - filtered_count} edges removed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
