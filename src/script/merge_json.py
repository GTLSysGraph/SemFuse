import json
import os

def merge_json_files(input_file_path, output_file_path, merged_file_path):
    """
    Merge two JSON files based on the same ID field
    
    Args:
        input_file_path: Path to the first JSON file (contains cosine_similarity and node_pair fields)
        output_file_path: Path to the second JSON file (contains output field)
        merged_file_path: Output path for the merged JSON file
    """

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"First file contains {len(input_data)} records")
    except FileNotFoundError:
        print(f"Error: File not found - {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: {input_file_path} is not a valid JSON file")
        return
    
    # Read the second JSON file
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        print(f"Second file contains {len(output_data)} records")
    except FileNotFoundError:
        print(f"Error: File not found - {output_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: {output_file_path} is not a valid JSON file")
        return
    
    output_dict = {item['id']: item for item in output_data}
    merged_data = []
    matched_count = 0
    
    for input_item in input_data:
        item_id = input_item['id']
        if item_id in output_dict:
            output_item = output_dict[item_id]
            common_fields = ['instruction', 'input']
            fields_match = True
            
            for field in common_fields:
                if field in input_item and field in output_item:
                    if input_item[field] != output_item[field]:
                        print(f"Warning: Field '{field}' mismatch for ID {item_id}")
                        fields_match = False
            
            # Merge data
            merged_item = {
                'id': item_id,
                'instruction': input_item.get('instruction', ''),
                'input': input_item.get('input', ''),
                'node_pair': input_item.get('node_pair', ''),
                'cosine_similarity': input_item.get('cosine_similarity', ''),
                'output': output_item.get('output', '')
            }
            
            merged_data.append(merged_item)
            matched_count += 1
        else:
            print(f"Warning: ID {item_id} not found in the second file")
    
    # Save merged data
    try:
        os.makedirs(os.path.dirname(merged_file_path), exist_ok=True)
        
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully matched and merged {matched_count} records")
        print(f"Merged file saved to: {merged_file_path}")
    except Exception as e:
        print(f"Error: Failed to save file - {str(e)}")

def main():
    # Define file paths
    input_file = "../data/input_json/Cornell_cosine_similarity.json"
    output_file = "../data/output_json/Cornell_cosine_similarity_cleaned.json"
    merged_file = "../data/merged_json/Cornell_cosine_similarity_merged.json"
    
    print("Starting JSON file merging...")
    print(f"Input file 1: {input_file}")
    print(f"Input file 2: {output_file}")
    print(f"Output file: {merged_file}")
    print("-" * 50)
    
    # Execute merging
    merge_json_files(input_file, output_file, merged_file)

if __name__ == "__main__":
    main()