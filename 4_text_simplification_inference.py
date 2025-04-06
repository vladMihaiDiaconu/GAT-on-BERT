import os
import json
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Import model definitions
import importlib
try:
    gat_module = importlib.import_module("3_gat_train_evaluate_model")
    TextSimplificationGAT = gat_module.TextSimplificationGAT
except ImportError:
    print("Error: Could not import the GAT model class. Make sure the training script is accessible.")
    raise

def load_model(model_path, config):
    """
    Load a trained model from a checkpoint file and auto-detect all architectural parameters.
    """
    print(f"Loading model from {model_path}")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    # Detect the hidden dimension from the state dict
    if 'output_layer.0.weight' in state_dict:
        detected_hidden_dim = state_dict['output_layer.0.weight'].shape[0]
        print(f"Detected hidden dimension: {detected_hidden_dim}")
        config['hidden_dim'] = detected_hidden_dim
    
    # Detect number of attention heads
    if 'conv1.att_src' in state_dict:
        detected_heads = state_dict['conv1.att_src'].shape[1]
        print(f"Detected number of attention heads: {detected_heads}")
        config['heads'] = detected_heads
    
    # For dropout, we can't detect it from the state dict as it's not a parameter
    # We'll keep the default value or the provided one
    
    # Create a model with the detected dimensions
    model = TextSimplificationGAT(
        in_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        out_dim=config['embedding_dim'],
        heads=config['heads'],
        dropout=config['dropout'],
        target_level=config['target_level']
    )
    
    # Load the state dictionary
    model.load_state_dict(state_dict)
    
    print(f"Model loaded with auto-detected architecture:")
    print(f"  - Hidden dimension: {config['hidden_dim']}")
    print(f"  - Attention heads: {config['heads']}")
    print(f"  - Dropout rate: {config['dropout']} (default/provided value)")
    
    return model

def simplify_text_embedding(model, embedding, device):
    """
    Apply the simplification model to a text embedding.
    """
    model = model.to(device)
    model.eval()
    
    # Convert input to tensor
    x = torch.tensor(embedding.reshape(1, -1), dtype=torch.float).to(device)
    
    # Create empty edge index (single node)
    edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
    
    # Forward pass
    with torch.no_grad():
        simplified_embedding = model(x, edge_index)
    
    return simplified_embedding.cpu().numpy().flatten()

def process_document_batch(model, documents, source_level, target_level, device):
    """
    Process a batch of documents and return their simplified embeddings.
    """
    model = model.to(device)
    model.eval()
    
    results = []
    
    for doc in tqdm(documents, desc=f"Simplifying {source_level} to {target_level}"):
        # Extract source embedding
        source_embedding = np.array(doc[source_level])
        
        # Ground truth target embedding (if available)
        if target_level in doc:
            target_embedding = np.array(doc[target_level])
        else:
            target_embedding = None
        
        # Simplify the embedding
        simplified_embedding = simplify_text_embedding(model, source_embedding, device)
        
        # Calculate similarity with target (if available)
        if target_embedding is not None:
            similarity = cosine_similarity([simplified_embedding], [target_embedding])[0][0]
            source_target_sim = cosine_similarity([source_embedding], [target_embedding])[0][0]
            improvement = similarity - source_target_sim
        else:
            similarity = None
            source_target_sim = None
            improvement = None
        
        # Store results
        result = {
            'id': doc.get('id', 'unknown'),
            'filename': doc.get('filename', 'unknown'),
            'source_embedding': source_embedding,
            'simplified_embedding': simplified_embedding,
            'target_embedding': target_embedding,
            'similarity': similarity,
            'source_target_similarity': source_target_sim,
            'improvement': improvement
        }
        
        results.append(result)
    
    return results

def save_results(results, output_dir, target_level):
    """
    Save the simplification results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for documents with target embeddings
    metrics = []
    for result in results:
        if result['similarity'] is not None:
            metrics.append({
                'id': result['id'],
                'filename': result['filename'],
                'similarity': result['similarity'],
                'source_target_similarity': result['source_target_similarity'],
                'improvement': result['improvement']
            })
    
    if metrics:
        # Create metrics dataframe and save
        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(output_dir, f'simplification_metrics_{target_level}.csv')
        metrics_df.to_csv(metrics_file, index=False)
        
        # Calculate summary statistics
        avg_similarity = np.mean(metrics_df['similarity'])
        avg_source_target = np.mean(metrics_df['source_target_similarity'])
        avg_improvement = np.mean(metrics_df['improvement'])
        
        print(f"\nSimplification Metrics ({target_level}):")
        print(f"Average similarity with target: {avg_similarity:.4f}")
        print(f"Average source-target similarity: {avg_source_target:.4f}")
        print(f"Average improvement: {avg_improvement:.4f}")
        
        # Create histogram of similarities
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_df['similarity'], bins=20, alpha=0.7, label='Predicted vs Target')
        plt.hist(metrics_df['source_target_similarity'], bins=20, alpha=0.7, label='Source vs Target')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.title(f'Similarity Distribution for {target_level.capitalize()} Simplification')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'similarity_histogram_{target_level}.png'))
    
    # Save the full results (including embeddings)
    full_results = []
    for result in results:
        # Convert numpy arrays to lists for JSON serialization
        result_copy = result.copy()
        result_copy['source_embedding'] = result_copy['source_embedding'].tolist()
        result_copy['simplified_embedding'] = result_copy['simplified_embedding'].tolist()
        if result_copy['target_embedding'] is not None:
            result_copy['target_embedding'] = result_copy['target_embedding'].tolist()
        
        full_results.append(result_copy)
    
    full_results_file = os.path.join(output_dir, f'simplification_results_{target_level}.json')
    with open(full_results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"Results saved to {output_dir}")


parser = argparse.ArgumentParser(description='Apply text simplification model to BERT embeddings')
parser.add_argument('--input', required=True, help='Input JSON file with BERT embeddings')
parser.add_argument('--output', default='simplification_output', help='Output directory')

args = parser.parse_args()

# Load input data
print(f"Loading data from {args.input}")
with open(args.input, 'r') as f:
    data = json.load(f)

# Convert to list if it's a dictionary
if isinstance(data, dict):
    data = [data]

print(f"Loaded {len(data)} documents")

# Get embedding dimension from first item
embedding_dim = len(data[0]['advanced'])
print(f"Embedding dimension: {embedding_dim}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model configurations
elem_config = {
    'embedding_dim': embedding_dim,
    'hidden_dim': 256,  # Default but adjustable based on how you decide to save the model
    'heads': 4,
    'dropout': 0.2,
    'target_level': 'elementary'
}

inter_config = {
    'embedding_dim': embedding_dim,
    'hidden_dim': 256,  # Default but adjustable based on how you decide to save the model
    'heads': 4,
    'dropout': 0.2,
    'target_level': 'intermediate'
}

# Load models if the files exist
if os.path.exists("model_advanced_to_elementary.pt"):
    print("\n=== Running Elementary Text Simplification ===")
    elem_model = load_model("model_advanced_to_elementary.pt", elem_config)
    
    # Process documents with elementary model
    elem_results = process_document_batch(elem_model, data, 'advanced', 'elementary', device)
    
    # Save elementary results
    save_results(elem_results, args.output, 'elementary')
else:
    print("Warning: Elementary model file model_advanced_to_elementary.pt not found")

if os.path.exists("model_advanced_to_intermediate.pt"):
    print("\n=== Running Intermediate Text Simplification ===")
    inter_model = load_model("model_advanced_to_intermediate.pt", inter_config)
    
    # Process documents with intermediate model
    inter_results = process_document_batch(inter_model, data, 'advanced', 'intermediate', device)
    
    # Save intermediate results
    save_results(inter_results, args.output, 'intermediate')
else:
    print("Warning: Intermediate model file model_advanced_to_intermediate.pt not found")

print("\nDone!")