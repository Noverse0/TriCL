import random
from itertools import permutations
import os
import matplotlib.pyplot as plt
import numpy as np
import dgl
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs: list[Tensor], num_nodes: int, num_edges: int):
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index


def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)


def knn_graph_dgl(tensor, k):
    """
    Compute the k-nearest neighbor graph using DGL for a given tensor.
    
    Parameters:
    - tensor: The input tensor of shape (n_samples, n_features)
    - k: Number of neighbors to consider
    
    Returns:
    - g: DGLGraph where an edge indicates a neighbor relation between two samples
    """
    neigh = NearestNeighbors(n_neighbors=k+1).fit(tensor)  # +1 because a sample is its own nearest neighbor
    _, indices = neigh.kneighbors(tensor)
    
    src_nodes = np.repeat(np.arange(tensor.shape[0]), k)
    dst_nodes = indices[:, 1:].reshape(-1)  # skip the first column as it's the sample itself
    
    g = dgl.graph((src_nodes, dst_nodes))
    
    return g

def overlapping_score_dgl(x1, x2, k):
    """
    Compute the overlapping score between two tensors' k-nearest neighbor graphs using DGL.
    
    Parameters:
    - tensor1, tensor2: The input tensors each of shape (n_samples, n_features)
    - k: Number of neighbors to consider for the knn graph
    
    Returns:
    - score: Overlapping score between the two graphs
    """
    graph1 = dgl.knn_graph(x1, k)
    graph2 = dgl.knn_graph(x2, k)
    
    # 아래 줄에서 'scipy_fmt' 인자를 제거
    adjacency_matrix1 = graph1.adjacency_matrix().to_dense().numpy()
    adjacency_matrix2 = graph2.adjacency_matrix().to_dense().numpy()
    
    # intersection = np.sum(np.logical_and(adjacency_matrix1, adjacency_matrix2))
    intersection = np.sum(adjacency_matrix1 * adjacency_matrix2)
    denom = np.sum(adjacency_matrix1)
    
    score = intersection / denom if denom != 0 else 0
    
    return score

def plot_tsne(n, labels, dir="tsne_figure", file_name="tsne_plot.png", perplexity=30):
    """
    Plot t-SNE of the data and save to a file.
    
    Args:
        data: PyTorch Tensor or NumPy array of shape (num_samples, num_features)
        labels: PyTorch Tensor or NumPy array of shape (num_samples,) for color coding
        file_name: The name of the file to save the plot
        perplexity: t-SNE perplexity parameter
    """
    
    # Convert data to NumPy array if it's a tensor
    if isinstance(n, torch.Tensor):
        n = n.detach().cpu().numpy()
    
    # Convert labels to NumPy array if it's a tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=500, init='random', learning_rate=200.0).fit_transform(n)
    
    # Compute silhouette score
    s_score = silhouette_score(tsne, labels)
    
    # Get distinct colors for each label
    num_unique_labels = len(set(labels))
    cmap = plt.get_cmap('tab20', num_unique_labels)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap=cmap, s=30)
    plt.gca().set_facecolor('white')  # Set background color to white
    plt.grid(False)  # Remove grid

    # Adding the silhouette score to the plot
    plt.title(f"Silhouette score: {s_score:.3f}")

    # Save the plot
    plt.savefig(os.path.join(dir, file_name))
    plt.close()
    