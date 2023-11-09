import argparse
import random
import wandb
import yaml
from tqdm import tqdm
import numpy as np
import torch
import math
import time

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, f1_score
from scipy.stats import mode

from TriCL.loader import DatasetLoader
from TriCL.models import HyperEncoder, TriCL
from TriCL.utils import drop_features, drop_incidence, valid_node_edge_mask, hyperedge_index_masking
from TriCL.evaluation import linear_evaluation


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(model_type, num_negs):
    features, hyperedge_index = data.features, data.hyperedge_index
    num_nodes, num_edges = data.num_nodes, data.num_edges

    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # Hypergraph Augmentation
    hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
    hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
    x1 = drop_features(features, params['drop_feature_rate'])
    x2 = drop_features(features, params['drop_feature_rate'])

    node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
    node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    node_mask = node_mask1 & node_mask2
    edge_mask = edge_mask1 & edge_mask2

    # Encoder
    n1, e1 = model(x1, hyperedge_index1, num_nodes, num_edges)
    n2, e2 = model(x2, hyperedge_index2, num_nodes, num_edges)
    
    # Projection Head
    n1, n2 = model.node_projection(n1), model.node_projection(n2)
    e1, e2 = model.edge_projection(e1), model.edge_projection(e2)
    
    loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
    if model_type in ['tricl_ng', 'tricl']:
        loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], params['tau_g'], batch_size=params['batch_size_1'], num_negs=num_negs)
    else:
        loss_g = 0
        
    if model_type in ['tricl']:
        masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask1)
        masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask2)
        loss_m1 = model.membership_level_loss(n1, e2[edge_mask2], masked_index2, params['tau_m'], batch_size=params['batch_size_2'])
        loss_m2 = model.membership_level_loss(n2, e1[edge_mask1], masked_index1, params['tau_m'], batch_size=params['batch_size_2'])
        loss_m = (loss_m1 + loss_m2) * 0.5
    else:
        loss_m = 0
    loss = loss_n + params['w_g'] * loss_g + params['w_m'] * loss_m
    loss.backward()
    optimizer.step()
    return loss.item()

def cluster_eval(embeddings, labels):
    num_clusters = labels.max().item() + 1
    kmeans = KMeans(n_clusters=num_clusters, n_init=20)
    labels_pred = kmeans.fit_predict(embeddings)
    
    # Calculate Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(labels, labels_pred)
    
    # Calculate F1 score
    # Find the permutation of labels that gives the best match
    labels_true_permuted = np.empty_like(labels_pred)
    for i in range(num_clusters):
        mask = (labels_pred == i)
        labels_true_permuted[mask] = mode(labels[mask])[0]
    
    f1 = f1_score(labels, labels_true_permuted, average='weighted')
    
    return nmi * 100, f1 * 100

def node_classification_eval(model, data, params, num_splits=20):
    model.eval()
    n, _ = model(data.features, data.hyperedge_index)
    
    lr = params['eval_lr']
    max_epoch = params['eval_epochs']
    
    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch))
        
    # nmi, f1
    n = n.cpu().detach().numpy()
    labels = data.labels.cpu().detach().numpy()
    nmi, f1 = cluster_eval(n, labels)
    
    return accs, nmi, f1



if __name__ == '__main__':
    wandb.init(project="sim_hgcl", mode="online")
    start = time.time()
    math.factorial(100000)
    parser = argparse.ArgumentParser('TriCL unsupervised learning.')
    parser.add_argument('--dataset', type=str, default='NTU2012', 
        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor', 
                 'zoo', '20newsW100', 'Mushroom', 'NTU2012', 'ModelNet40'])
    parser.add_argument('--model_type', type=str, default='tricl', choices=['tricl_n', 'tricl_ng', 'tricl'])
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    print(params)

    data = DatasetLoader().load(args.dataset).to(args.device)

    accs ,nmis, f1s = [], [], []
    gpu_memory_allocated = []
    gpu_max_memory_allocated = []
    for seed in range(args.num_seeds):
        fix_seed(seed)
        encoder = HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        model = TriCL(encoder, params['proj_dim']).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        wandb.watch(model, log='all')
        for epoch in tqdm(range(1, params['epochs'] + 1)):
            loss = train(args.model_type, num_negs=None)
            wandb.log({"Epoch": epoch, "Loss": loss})
        
        # At the end of each seed iteration, record the memory usage
        gpu_memory_allocated.append(torch.cuda.memory_allocated(args.device))
        gpu_max_memory_allocated.append(torch.cuda.max_memory_allocated(args.device))
        
        # evaluation
        acc, nmi, f1 = node_classification_eval(model, data, params)
        accs.append(acc)
        nmis.append(nmi)
        f1s.append(f1)

        accs.append(acc)
        acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
        print(f'seed: {seed}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
            f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}')
        print(f'Clustering NMI: {nmi:.2f}, F1: {f1:.2f}')

    accs = np.array(accs).reshape(-1, 3)
    accs_mean = list(np.mean(accs, axis=0))
    accs_std = list(np.std(accs, axis=0))
    print(f'[Final] dataset: {args.dataset}, test_acc: {accs_mean[2]:.2f}+-{accs_std[2]:.2f}')
    print(f'Clustering NMI: {np.mean(nmis, axis=0):.2f}, F1: {np.mean(f1s, axis=0):.2f}')
    
    # Calculate and print the average GPU memory usage
    avg_gpu_memory_allocated = sum(gpu_memory_allocated) / len(gpu_memory_allocated)
    avg_gpu_max_memory_allocated = sum(gpu_max_memory_allocated) / len(gpu_max_memory_allocated)
    print(f'Average allocated GPU memory: {avg_gpu_memory_allocated / (1024 ** 2):.2f} MB')
    print(f'Average peak allocated GPU memory: {avg_gpu_max_memory_allocated / (1024 ** 2):.2f} MB')    
    
    end = time.time()
    print(f"{(end - start)/5:.2f} sec")
    
    wandb.log({
        "dataset": args.dataset,
        "Train_Acc_Mean": accs_mean[0],
        "Train_Acc_Std": accs_std[0],
        "Valid_Acc_Mean": accs_mean[1],
        "Valid_Acc_Std": accs_std[1],
        "Test_Acc_Mean": accs_mean[2],
        "Test_Acc_Std": accs_std[2],
        "NMI_Mean": np.mean(nmis, axis=0),
        "F1_Mean": np.mean(f1s, axis=0),
        "Time": (end - start)/5,
        "Average_GPU_Memory_Allocated_MB": avg_gpu_memory_allocated / (1024 ** 2),
        "Average_GPU_Peak_Memory_Allocated_MB": avg_gpu_max_memory_allocated / (1024 ** 2),
        "model": 'TriCL',
        "gpu": "2080TI"
    })