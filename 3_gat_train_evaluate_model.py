import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
EMBEDDING_DIM = 384  # Dimension of BERT embeddings
HIDDEN_DIM = 256    # Hidden dimension for GAT
OUTPUT_DIM = 384    # Output dimension (same as input for reconstruction)
NUM_HEADS = 4       # Number of attention heads
DROPOUT = 0.2       # Dropout rate
EPOCHS = 100        # Training epochs
LR = 0.001          # Learning rate

class TextSimplificationGAT(nn.Module):
    """
    Graph Attention Network for text simplification.
    Transforms advanced text embeddings to elementary or intermediate text embeddings.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.2, target_level='elementary'):
        super(TextSimplificationGAT, self).__init__()
        self.target_level = target_level
        
        # First GAT layer with multiple attention heads
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        
        # Second GAT layer that combines the heads
        self.conv2 = GATConv(hidden_dim, hidden_dim, dropout=dropout)
        
        # Output transformation to match target embedding dimension
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x, edge_index):
        # Apply first GAT layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Apply second GAT layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Transform to output dimension
        x = self.output_layer(x)
        
        return x

def load_data(json_file):
    """
    Load and prepare data from a JSON file containing text embeddings.
    """
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract embeddings
    advanced_embeddings = []
    intermediate_embeddings = []
    elementary_embeddings = []
    ids = []
    filenames = []
    
    # Iterating through the data
    for item in data:
        ids.append(item['id'])
        filenames.append(item['filename'])
        advanced_embeddings.append(np.array(item['advanced']))
        intermediate_embeddings.append(np.array(item['intermediate']))
        elementary_embeddings.append(np.array(item['elementary']))
    
    return {
        'ids': ids,
        'filenames': filenames,
        'advanced': np.array(advanced_embeddings),
        'intermediate': np.array(intermediate_embeddings),
        'elementary': np.array(elementary_embeddings)
    }

def create_graph_data(data, level_pairs, test_size=0.2):
    """
    Create PyTorch Geometric graph data objects for training.
    
    Args:
        data: Dictionary containing embeddings
        level_pairs: List of (source, target) pairs, e.g., [('advanced', 'elementary')]
        test_size: Fraction of data to use for testing
    
    Returns:
        train_loader, test_loader
    """
    train_data_list = []
    test_data_list = []
    
    for source_level, target_level in level_pairs:
        source_embeddings = data[source_level]
        target_embeddings = data[target_level]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            source_embeddings, target_embeddings, test_size=test_size, random_state=42
        )
        
        # Create graph data objects for training set
        for i in range(len(X_train)):
            # For each document, create a small graph where each node is a token/word
            # Here, we're creating a fully connected graph for simplicity
            num_nodes = 1  # We're treating each embedding as a single node for simplicity
            
            # Node features (source embedding)
            x = torch.tensor(X_train[i].reshape(num_nodes, -1), dtype=torch.float)
            
            # Create fully connected edges (for a single node, this is empty)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long) if num_nodes > 1 else torch.zeros((2, 0), dtype=torch.long)
            
            # Target embedding
            y = torch.tensor(y_train[i].reshape(num_nodes, -1), dtype=torch.float)
            
            # Create Data object
            graph_data = Data(x=x, edge_index=edge_index, y=y)
            train_data_list.append(graph_data)
        
        # Create graph data objects for test set
        for i in range(len(X_test)):
            num_nodes = 1
            x = torch.tensor(X_test[i].reshape(num_nodes, -1), dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long) if num_nodes > 1 else torch.zeros((2, 0), dtype=torch.long)
            y = torch.tensor(y_test[i].reshape(num_nodes, -1), dtype=torch.float)
            graph_data = Data(x=x, edge_index=edge_index, y=y)
            test_data_list.append(graph_data)
    
    # Create data loaders
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, device, epochs=100, lr=0.001):
    """
    Train the GAT model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index)
            
            # Calculate loss
            loss = criterion(out, batch.y)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch.num_graphs
        
        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                test_loss += loss.item() * batch.num_graphs
        
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Test Loss for {model.target_level} Level')
    plt.legend()
    plt.savefig(f'loss_curve_{model.target_level}.png')
    
    return model, train_losses, test_losses


# Load data
data = load_data('onestopenglish_aligned_bert_embeddings.json')

# Create graph data for advanced → elementary
train_loader_elem, test_loader_elem = create_graph_data(
    data, [('advanced', 'elementary')]
)

# Create graph data for advanced → intermediate
train_loader_inter, test_loader_inter = create_graph_data(
    data, [('advanced', 'intermediate')]
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize and train model for elementary level
model_elem = TextSimplificationGAT(
    in_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    out_dim=OUTPUT_DIM,
    heads=NUM_HEADS,
    dropout=DROPOUT,
    target_level='elementary'
)

print("Training model for Advanced → Elementary")
model_elem, train_losses_elem, test_losses_elem = train_model(
    model_elem, train_loader_elem, test_loader_elem, device, EPOCHS, LR
)

# Save the trained model
torch.save(model_elem.state_dict(), 'model_advanced_to_elementary.pt')

# Initialize and train model for intermediate level
model_inter = TextSimplificationGAT(
    in_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    out_dim=OUTPUT_DIM,
    heads=NUM_HEADS,
    dropout=DROPOUT,
    target_level='intermediate'
)

print("Training model for Advanced → Intermediate")
model_inter, train_losses_inter, test_losses_inter = train_model(
    model_inter, train_loader_inter, test_loader_inter, device, EPOCHS, LR
)

# Save the trained model
torch.save(model_inter.state_dict(), 'model_advanced_to_intermediate.pt')

print("Training completed and models saved.")