import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

def truncate_text(text, max_tokens=1000):
    tokens = text.split()
    return ' '.join(tokens[:max_tokens]) if len(tokens) > max_tokens else text

def build_prompt_from_cosine_similarity(npz_path, save_path, dataset_name, top_k=10):
    data = np.load(npz_path, allow_pickle=True)
    
    node_texts = data['node_texts']
    node_features = data['node_features']
    edges = data['edges']
    
    # Build a set of existing edges for fast lookup
    existing_edges = set()
    for i, j in zip(edges[0], edges[1]):
        existing_edges.add((i, j))
        existing_edges.add((j, i))
    
    if dataset_name.lower() == "cornell":
        instruction = (
            "You are an expert in analyzing the relevance between nodes in a graph neural network. "
            f"In the WebKB dataset ({dataset_name}), each node represents a webpage from a university, "
            "and edges represent hyperlinks. Each webpage belongs to one of five categories: "
            "student, staff, faculty, course, or project. Given the text information of two webpages, "
            "analyze how likely they belong to the same category. Your analysis should be around 100 words. "
            "Your response should be formatted in JSON, with two keys: \"Analysis\" for your written analysis, "
            "and \"Relevance Score\" (an integer from 1 to 5, where 5 means very likely the same category, "
            "and 1 means completely unrelated)."
        )
    elif dataset_name.lower() == "texas":
        instruction = (
            "You are an expert in analyzing the relevance between nodes in a graph neural network. "
            f"In the WebKB dataset ({dataset_name}), each node represents a webpage from a university, "
            "and edges represent hyperlinks. Each webpage belongs to one of five categories: "
            "student, staff, faculty, course, or project. Given the text information of two webpages, "
            "analyze how likely they belong to the same category. Your analysis should be around 100 words. "
            "Your response should be formatted in JSON, with two keys: \"Analysis\" for your written analysis, "
            "and \"Relevance Score\" (an integer from 1 to 5, where 5 means very likely the same category, "
            "and 1 means completely unrelated)."
        )
    elif dataset_name.lower() == "wisconsin":
        instruction = (
            "You are an expert in analyzing the relevance between nodes in a graph neural network. "
            f"In the WebKB dataset ({dataset_name}), each node represents a webpage from a university, "
            "and edges represent hyperlinks. Each webpage belongs to one of five categories: "
            "student, staff, faculty, course, or project. Given the text information of two webpages, "
            "analyze how likely they belong to the same category. Your analysis should be around 100 words. "
            "Your response should be formatted in JSON, with two keys: \"Analysis\" for your written analysis, "
            "and \"Relevance Score\" (an integer from 1 to 5, where 5 means very likely the same category, "
            "and 1 means completely unrelated)."
        )
    elif dataset_name.lower() == "actor":
        instruction = (
            "You are an expert in analyzing the relevance between nodes in a graph neural network. "
            "In the Actor dataset, each node represents an actor extracted from Wikipedia. "
            "An edge between two nodes indicates that the corresponding actors co-occur on the same Wikipedia page. "
            "Each actor is assigned to one of five categories: "
            "1) actors labeled only as 'American film actors', "
            "2) actors labeled as both 'American film actors' and 'American television actors', "
            "3) actors labeled as both 'American television actors' and 'American stage actors', "
            "4) 'English actors' (including film, television, or stage), and "
            "5) 'Canadian actors'. "
            "Given the Wikipedia-based textual description of two actors, analyze how likely they belong to the same category. "
            "Your analysis should be around 100 words. "
            "Your response should be formatted in JSON, with two keys: \"Analysis\" for your written analysis, "
            "and \"Relevance Score\" (an integer from 1 to 5, where 5 means very likely the same category, and 1 means completely unrelated)."
        )
    elif dataset_name.lower() == "amazon":
        instruction = (
            "You are an expert in analyzing the similarity and relevance between products in an e-commerce co-purchasing network.\n"
            f"In the Amazon dataset ({dataset_name}), each node represents a product such as a book, DVD, or music CD, "
            "and edges represent products that are frequently bought together. Each product is associated with a collection of user reviews.\n\n"

            "Each product's textual description contains the following structured information:\n"
            "- 'Name/title of the product': The product's name.\n"
            "- 'Amazon Salesrank': The product's sales ranking.\n"
            "- 'Total number of co-purchased products': The number of other products frequently bought with it.\n"
            "- 'Number of reviews': Total number of user reviews.\n"
            "- 'The first five reviews': These contain up to five of the earliest reviews for this product, each in the format:\n"
            "  'rating: X votes: Y helpful: Z' where:\n"
            "    • 'rating' is the score (1 to 5) that a user gave the product,\n"
            "    • 'votes' is the number of users who voted on the review,\n"
            "    • 'helpful' is how many users marked the review as helpful.\n\n"

            "Your task is: Given the textual descriptions of two products, analyze how likely it is that they belong to the same rating group.\n"
            "The target classification has five classes based on average product rating:\n"
            "- Rating 5.0\n"
            "- Rating 4.5\n"
            "- Rating 4.0\n"
            "- Rating 3.5\n"
            "- Rating lower than 3.5\n\n"

            "You should write an analysis of about 100 words that explains whether the two products are likely to belong to the same rating category.\n"
            "Your response must be formatted as a JSON object with the following two fields:\n"
            "1. 'Analysis': your explanation of the similarity or difference.\n"
            "2. 'Relevance Score': an integer from 1 to 5:\n"
            "    • 5 means they are very likely in the same category,\n"
            "    • 1 means they are completely unrelated.\n"
        )
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    # Compute cosine similarity for all node pairs
    print("Computing cosine similarity matrix...")
    similarity_matrix = cosine_similarity(node_features)
    results = []
    id_counter = 0
    
    # For each node, find top-k most similar nodes
    num_nodes = len(node_texts)
    for i in range(num_nodes):
        similarities = similarity_matrix[i]
        node_sim_pairs = [(j, similarities[j]) for j in range(num_nodes) if j != i]
        node_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = []
        for j, sim_score in node_sim_pairs:
            if (i, j) not in existing_edges and len(selected_nodes) < top_k:
                selected_nodes.append((j, sim_score))

        if len(selected_nodes) < top_k:
            print(f"Warning: Node {i} only has {len(selected_nodes)} valid candidates (target: {top_k})")

        for j, sim_score in selected_nodes:
            input_text = (
                f"Node 1:\n{truncate_text(node_texts[i])}\n\n"
                f"Node 2:\n{truncate_text(node_texts[j])}"
            )
            results.append({
                "id": str(id_counter),
                "instruction": instruction,
                "input": input_text,
                "node_pair": [int(i), int(j)], 
                "cosine_similarity": float(sim_score)  
            })
            id_counter += 1
    
    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(results)} cosine similarity-based samples to: {save_path}")
    print(f"Total nodes processed: {num_nodes}")
    print(f"Average samples per node: {len(results)/num_nodes:.2f}")
    
    return results

if __name__ == "__main__":
    build_prompt_from_cosine_similarity(
        npz_path="../../dataset/Cornell_filtered.npz",
        save_path="../data/input_json/Cornell_cosine_similarity.json",
        dataset_name="Cornell",
        top_k=5
    )
