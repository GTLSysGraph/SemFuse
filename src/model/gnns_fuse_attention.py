from __future__ import annotations
import argparse
import os
import math
import numpy as np
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ==================== Utility Functions ====================

def add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Add self-loops to edge indices"""
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    return torch.cat([edge_index, self_loops], dim=1)


def degree(index: Tensor, num_nodes: int, device=None, dtype=torch.float32) -> Tensor:
    """Compute node in-degree"""
    if device is None:
        device = index.device
    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    one = torch.ones(index.size(0), device=device, dtype=dtype)
    deg.index_add_(0, index, one)
    return deg


# ==================== Graph Neural Network Layers ====================

class GCNLayer(nn.Module):
    """GCN layer implementation"""
    def __init__(self, in_channels: int, out_channels: int, add_self_loops_flag: bool = True, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.add_self_loops_flag = add_self_loops_flag
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lin.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def _normalized_edge_weight(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        src, dst = edge_index
        deg_dst = degree(dst, num_nodes, device=edge_index.device)
        deg_src = degree(src, num_nodes, device=edge_index.device)
        deg_inv_sqrt_dst = (deg_dst + 1e-12).pow(-0.5)
        deg_inv_sqrt_src = (deg_src + 1e-12).pow(-0.5)
        norm = deg_inv_sqrt_src[src] * deg_inv_sqrt_dst[dst]
        return norm

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        if num_nodes is None:
            num_nodes = x.size(0)
        if self.add_self_loops_flag:
            edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index
        xW = self.lin(x)
        norm = self._normalized_edge_weight(edge_index, num_nodes)

        out = torch.zeros(num_nodes, xW.size(1), device=x.device, dtype=xW.dtype)
        messages = xW[src] * norm.unsqueeze(-1)
        out.index_add_(0, dst, messages)

        if self.bias is not None:
            out = out + self.bias
        return out


class GATLayer(nn.Module):
    """GAT layer implementation"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops_flag: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops_flag = add_self_loops_flag

        self.lin = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(heads, out_channels))

        if bias:
            bias_dim = out_channels * heads if concat else out_channels
            self.bias = nn.Parameter(torch.zeros(bias_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight, gain=nn.init.calculate_gain("leaky_relu", self.negative_slope))
        nn.init.xavier_uniform_(self.att_src, gain=nn.init.calculate_gain("leaky_relu", self.negative_slope))
        nn.init.xavier_uniform_(self.att_dst, gain=nn.init.calculate_gain("leaky_relu", self.negative_slope))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        if num_nodes is None:
            num_nodes = x.size(0)
        if self.add_self_loops_flag:
            edge_index = add_self_loops(edge_index, num_nodes)

        src, dst = edge_index

        h = self.lin(x)
        h = h.view(-1, self.heads, self.out_channels)

        h_src = h[src]
        h_dst = h[dst]

        e_src = (h_src * self.att_src).sum(dim=-1)
        e_dst = (h_dst * self.att_dst).sum(dim=-1)
        e = F.leaky_relu(e_src + e_dst, negative_slope=self.negative_slope)

        exp_e = torch.exp(e)
        denom = torch.zeros(num_nodes, self.heads, device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-16)

        if self.training and self.dropout > 0.0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        messages = h_src * alpha.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, messages)

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out


class SAGEConv(nn.Module):
    """GraphSAGE layer implementation"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
        if not bias:
            with torch.no_grad():
                self.lin_self.bias.zero_()
            self.lin_self.bias.requires_grad_(False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lin_neigh.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lin_self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lin_self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        if num_nodes is None:
            num_nodes = x.size(0)

        src, dst = edge_index

        neigh_sum = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        neigh_sum.index_add_(0, dst, x[src])

        deg = degree(dst, num_nodes, device=x.device, dtype=x.dtype).clamp(min=1.0).unsqueeze(-1)
        neigh_mean = neigh_sum / deg

        out = self.lin_neigh(neigh_mean) + self.lin_self(x)
        return out


# ==================== Encoder Models ====================

class GCNEncoder(nn.Module):
    """GCN encoder - returns embeddings instead of logits"""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        add_self_loops: bool = True,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, "Encoder requires at least 1 layer."
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        if num_layers == 1:
            self.layers.append(GCNLayer(in_channels, hidden_channels, add_self_loops_flag=add_self_loops))
        else:
            self.layers.append(GCNLayer(in_channels, hidden_channels, add_self_loops_flag=add_self_loops))
            if self.bns is not None:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            for _ in range(num_layers - 1):
                self.layers.append(GCNLayer(hidden_channels, hidden_channels, add_self_loops_flag=add_self_loops))
                if self.bns is not None:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """GAT encoder - returns embeddings instead of logits"""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
        attn_dropout: float = 0.6,
        add_self_loops: bool = True,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, "Encoder requires at least 1 layer."
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        if num_layers == 1:
            self.layers.append(
                GATLayer(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    concat=False,  # average heads in the last layer
                    dropout=attn_dropout,
                    add_self_loops_flag=add_self_loops,
                )
            )
        else:
            self.layers.append(
                GATLayer(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=attn_dropout,
                    add_self_loops_flag=add_self_loops,
                )
            )
            if self.bns is not None:
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

            for i in range(num_layers - 2):
                self.layers.append(
                    GATLayer(
                        hidden_channels * heads,
                        hidden_channels,
                        heads=heads,
                        concat=True,
                        dropout=attn_dropout,
                        add_self_loops_flag=add_self_loops,
                    )
                )
                if self.bns is not None:
                    self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

            # last layer outputs hidden_channels dimension (no concat)
            self.layers.append(
                GATLayer(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    dropout=attn_dropout,
                    add_self_loops_flag=add_self_loops,
                )
            )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder - returns embeddings instead of logits"""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, "Encoder requires at least 1 layer."
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        if num_layers == 1:
            self.layers.append(SAGEConv(in_channels, hidden_channels))
        else:
            self.layers.append(SAGEConv(in_channels, hidden_channels))
            if self.bns is not None:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden_channels, hidden_channels))
                if self.bns is not None:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ==================== Multi-view Attention Module ====================

class MultiViewAttention(nn.Module):
    """Multi-view attention fusion module"""
    def __init__(self, in_feats, num_views=2, hidden=64):
        super(MultiViewAttention, self).__init__()
        self.num_views = num_views
        self.attn_fc = nn.ModuleList([nn.Linear(in_feats, hidden) for _ in range(num_views)])
        self.context_vec = nn.Parameter(torch.randn(num_views, hidden))

    def forward(self, node_feats_views):
        """
        node_feats_views: list of [N, F] represents node features for each view
        Returns: (fused_feats, attn_weights)
        """
        attn_scores = []
        for v in range(self.num_views):
            h_v = torch.tanh(self.attn_fc[v](node_feats_views[v]))
            score_v = (h_v * self.context_vec[v]).sum(dim=-1, keepdim=True)
            attn_scores.append(score_v)
        attn_scores = torch.cat(attn_scores, dim=1)
        attn_weights = F.softmax(attn_scores, dim=1)
        fused_feats = sum(attn_weights[:, v:v+1] * node_feats_views[v] for v in range(self.num_views))
        return fused_feats, attn_weights


# ==================== Loss Functions ====================

def compute_losses(logits, labels, masks, feats_views, attn_weights, edge_indices, 
                   lambda_smooth=0.01, lambda_view=0.005, lambda_entropy=0.005):
    """Compute multi-task loss"""
    # Classification loss
    loss_cls = F.cross_entropy(logits[masks['train']], labels[masks['train']])

    # Node smoothness loss
    loss_smooth = 0
    for i, feats in enumerate(feats_views):
        edge_idx = edge_indices[i]
        if edge_idx.size(1) > 0:
            loss_smooth += torch.mean((feats[edge_idx[0]] - feats[edge_idx[1]]).pow(2))

    # View consistency loss
    loss_view_consistency = 0
    for i in range(len(feats_views)):
        for j in range(i + 1, len(feats_views)):
            cosine_sim = F.cosine_similarity(feats_views[i], feats_views[j], dim=-1)
            loss_view_consistency += -torch.mean(cosine_sim)

    # Attention entropy loss
    entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=1).mean()
    loss_entropy = -entropy

    # Total loss
    total_loss = loss_cls + lambda_smooth * loss_smooth + lambda_view * loss_view_consistency + lambda_entropy * loss_entropy

    return total_loss, loss_cls, loss_smooth, loss_view_consistency, loss_entropy


# ==================== Data Loading ====================

def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Convert a directed graph to an undirected graph"""
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
    return edge_index_undirected


def load_npz(path: str, device: str = "cpu", transpose_edges: bool = False, make_undirected: bool = False):
    """Load graph data from NPZ format"""
    data = np.load(path, allow_pickle=True)
    edges = data["edges"]
    node_features = data["node_features"]
    node_labels = data["node_labels"]
    train_masks = data["train_masks"]
    val_masks = data["val_masks"]
    test_masks = data["test_masks"]

    if transpose_edges or (edges.ndim == 2 and edges.shape[0] != 2 and edges.shape[1] == 2):
        edges = edges.T
    assert edges.shape[0] == 2, f"edges must be shape [2, E], got {edges.shape}"
    
    edge_index = torch.from_numpy(edges.astype(np.int64, copy=False))
    x = torch.from_numpy(node_features.astype(np.float32, copy=False))
    y = torch.from_numpy(node_labels.astype(np.int64, copy=False))
    train_mask = torch.from_numpy(train_masks.astype(bool, copy=False))
    val_mask = torch.from_numpy(val_masks.astype(bool, copy=False))
    test_mask = torch.from_numpy(test_masks.astype(bool, copy=False))

    if make_undirected:
        edge_index = to_undirected(edge_index)

    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    return x, edge_index, y, train_mask, val_mask, test_mask


# ==================== Model Construction ====================

def build_encoder(
    model_name: str,
    in_dim: int,
    hidden: int,
    layers: int,
    dropout: float,
    heads: int,
    attn_dropout: float,
    add_self_loops: bool,
    use_bn: bool,
):
    """Build encoder"""
    if model_name == "gcn":
        return GCNEncoder(
            in_channels=in_dim,
            hidden_channels=hidden,
            num_layers=layers,
            dropout=dropout,
            add_self_loops=add_self_loops,
            use_batchnorm=use_bn,
        )
    elif model_name == "gat":
        return GATEncoder(
            in_channels=in_dim,
            hidden_channels=hidden,
            num_layers=layers,
            heads=heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            add_self_loops=add_self_loops,
            use_batchnorm=use_bn,
        )
    else:  # sage
        return GraphSAGEEncoder(
            in_channels=in_dim,
            hidden_channels=hidden,
            num_layers=layers,
            dropout=dropout,
            use_batchnorm=use_bn,
        )


# ==================== Training and Evaluation ====================

def train_one_epoch(encoder_orig, encoder_clean, attn_module, classifier, 
                   x, edge_orig, edge_clean, y, train_mask, optimizer, args):
    """Train one epoch"""
    encoder_orig.train()
    encoder_clean.train()
    attn_module.train()
    classifier.train()
    
    optimizer.zero_grad()
    
    # Encode both views
    emb_orig = encoder_orig(x, edge_orig)
    emb_clean = encoder_clean(x, edge_clean)
    
    # Attention fusion
    fused_emb, attn_weights = attn_module([emb_orig, emb_clean])
    
    # Classification
    logits = classifier(fused_emb)
    
    # Compute losses
    masks = {'train': train_mask}
    edge_indices = [edge_orig, edge_clean]
    total_loss, loss_cls, loss_smooth, loss_view, loss_entropy = compute_losses(
        logits, y, masks, [emb_orig, emb_clean], attn_weights, edge_indices,
        lambda_smooth=args.lambda_smooth,
        lambda_view=args.lambda_view,
        lambda_entropy=args.lambda_entropy
    )
    
    total_loss.backward()
    optimizer.step()
    
    return float(total_loss.item()), float(loss_cls.item()), float(loss_smooth.item()), float(loss_view.item()), float(loss_entropy.item())


@torch.no_grad()
def eval_metrics(encoder_orig, encoder_clean, attn_module, classifier, 
                x, edge_orig, edge_clean, y, mask):
    """Evaluate model"""
    encoder_orig.eval()
    encoder_clean.eval()
    attn_module.eval()
    classifier.eval()
    
    emb_orig = encoder_orig(x, edge_orig)
    emb_clean = encoder_clean(x, edge_clean)
    fused_emb, _ = attn_module([emb_orig, emb_clean])
    logits = classifier(fused_emb)
    
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == y[mask]).sum().item()
    total = int(mask.sum().item())
    acc = correct / max(total, 1)
    return acc


def set_seed(seed: int):
    """Set random seed"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_seed(seed, args, x, edge_orig, edge_clean, y, train_mask, val_mask, test_mask, 
                   in_dim, num_classes, device):
    """Run experiment for a single seed"""
    print(f"\n{'='*60}")
    print(f"Running with seed {seed}")
    print(f"{'='*60}")
    set_seed(seed)
    
    # Build model
    encoder_orig = build_encoder(
        args.model, in_dim, args.hidden, args.layers, args.dropout,
        args.heads, args.attn_dropout, not args.no_self_loops, args.use_bn
    ).to(device)
    
    encoder_clean = build_encoder(
        args.model, in_dim, args.hidden, args.layers, args.dropout,
        args.heads, args.attn_dropout, not args.no_self_loops, args.use_bn
    ).to(device)
    
    attn_module = MultiViewAttention(in_feats=args.hidden, num_views=2, hidden=64).to(device)
    classifier = nn.Linear(args.hidden, num_classes).to(device)
    
    # Optimizer
    all_params = (list(encoder_orig.parameters()) + list(encoder_clean.parameters()) +
                  list(attn_module.parameters()) + list(classifier.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    
    best_val = 0.0
    best_test_at_val = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        loss, loss_cls, loss_smooth, loss_view, loss_entropy = train_one_epoch(
            encoder_orig, encoder_clean, attn_module, classifier,
            x, edge_orig, edge_clean, y, train_mask, optimizer, args
        )
        
        val_acc = eval_metrics(encoder_orig, encoder_clean, attn_module, classifier,
                               x, edge_orig, edge_clean, y, val_mask)
        test_acc = eval_metrics(encoder_orig, encoder_clean, attn_module, classifier,
                                x, edge_orig, edge_clean, y, test_mask)
        
        if val_acc > best_val:
            best_val = val_acc
            best_test_at_val = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss {loss:.4f} (cls:{loss_cls:.3f} sm:{loss_smooth:.3f} vw:{loss_view:.3f} ent:{loss_entropy:.3f}) | "
                  f"val {val_acc:.4f} | test {test_acc:.4f} | best_val {best_val:.4f} | best_test {best_test_at_val:.4f}")
    
    print(f"Seed {seed} finished. Best val: {best_val:.4f}, Test at best val: {best_test_at_val:.4f}")
    return best_val, best_test_at_val


# ==================== Main Function ====================

def parse_args():
    p = argparse.ArgumentParser(description="Multi-view GNN with attention fusion")
    
    # Data settings
    p.add_argument("--data-orig", type=str, required=True, help="Path to original graph .npz")
    p.add_argument("--data-clean", type=str, required=True, help="Path to cleaned graph .npz")
    p.add_argument("--transpose-edges", action="store_true", help="Transpose edges if [E,2]")
    p.add_argument("--no-make-undirected", action="store_false", dest="make_undirected", 
                   default=True, help="Disable making graph undirected")
    
    # Model settings
    p.add_argument("--model", type=str, default="sage", choices=["gcn", "gat", "sage"], 
                   help="Encoder type")
    p.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    p.add_argument("--layers", type=int, default=2, help="Number of encoder layers")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    p.add_argument("--heads", type=int, default=8, help="GAT heads")
    p.add_argument("--attn-dropout", type=float, default=0.6, help="GAT attention dropout")
    p.add_argument("--no-self-loops", action="store_true", help="Disable self-loops")
    p.add_argument("--use-bn", action="store_true", help="Use BatchNorm")
    
    # Training settings
    p.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    p.add_argument("--epochs", type=int, default=400, help="Training epochs")
    p.add_argument("--patience", type=int, default=200, help="Early stopping patience")
    p.add_argument("--log-every", type=int, default=10, help="Log every N epochs")
    
    # Loss weights
    p.add_argument("--lambda-smooth", type=float, default=0, help="Smoothness loss weight")
    p.add_argument("--lambda-view", type=float, default=0, help="View consistency loss weight")
    p.add_argument("--lambda-entropy", type=float, default=0.005, help="Entropy loss weight")
    
    # Experiment settings
    p.add_argument("--seeds", nargs='+', type=int, 
                   default=[76, 1, 68, 59, 24, 64, 17, 128, 119, 45], 
                   #default=[421, 76, 67, 88, 860, 70, 17, 128, 151, 39],
                   help="Random seeds for multiple runs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device: cpu or cuda")
    
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print(f"Multi-View Graph Neural Network Training")
    print(f"{'='*70}")
    print(f"Model: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"Original data: {args.data_orig}")
    print(f"Cleaned data: {args.data_clean}")
    
    # Load original graph data
    print(f"\nLoading original graph...")
    x_orig, edge_orig, y_orig, train_mask_orig, val_mask_orig, test_mask_orig = load_npz(
        args.data_orig, device=device, 
        transpose_edges=args.transpose_edges, 
        make_undirected=args.make_undirected
    )
    
    # Load cleaned graph data
    print(f"Loading cleaned graph...")
    x_clean, edge_clean, y_clean, train_mask_clean, val_mask_clean, test_mask_clean = load_npz(
        args.data_clean, device=device,
        transpose_edges=args.transpose_edges,
        make_undirected=args.make_undirected
    )
    
    # Validate data consistency
    assert torch.equal(x_orig, x_clean), "Node features must be identical"
    assert torch.equal(y_orig, y_clean), "Node labels must be identical"
    assert torch.equal(train_mask_orig, train_mask_clean), "Train masks must be identical"
    assert torch.equal(val_mask_orig, val_mask_clean), "Val masks must be identical"
    assert torch.equal(test_mask_orig, test_mask_clean), "Test masks must be identical"
    
    # Use features and labels from the original graph
    x = x_orig
    y = y_orig
    train_mask = train_mask_orig
    val_mask = val_mask_orig
    test_mask = test_mask_orig
    
    in_dim = int(x.size(1))
    num_classes = int(y.max().item() + 1)
    
    print(f"\nDataset statistics:")
    print(f"  Nodes: {x.size(0)}")
    print(f"  Features: {in_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Original edges: {edge_orig.size(1)}")
    print(f"  Cleaned edges: {edge_clean.size(1)}")
    print(f"  Edge difference: {edge_clean.size(1) - edge_orig.size(1)}")
    print(f"  Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    
    print(f"\nModel configuration:")
    print(f"  Hidden dim: {args.hidden}")
    print(f"  Layers: {args.layers}")
    print(f"  Dropout: {args.dropout}")
    if args.model == "gat":
        print(f"  GAT heads: {args.heads}")
        print(f"  Attention dropout: {args.attn_dropout}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"\nLoss weights:")
    print(f"  Smoothness: {args.lambda_smooth}")
    print(f"  View consistency: {args.lambda_view}")
    print(f"  Entropy: {args.lambda_entropy}")
    
    print(f"\nRunning experiments with seeds: {args.seeds}")
    
    # Run experiments across multiple seeds
    all_val_accs = []
    all_test_accs = []
    
    for seed in args.seeds:
        val_acc, test_acc = run_single_seed(
            seed, args, x, edge_orig, edge_clean, y, 
            train_mask, val_mask, test_mask,
            in_dim, num_classes, device
        )
        all_val_accs.append(val_acc)
        all_test_accs.append(test_acc)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary across all seeds")
    print(f"{'='*70}")
    print(f"Val accuracy:  {np.mean(all_val_accs):.4f} ± {np.std(all_val_accs):.4f}")
    print(f"Test accuracy: {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}")
    print(f"\nIndividual results:")
    for i, seed in enumerate(args.seeds):
        print(f"  Seed {seed:4d}: val {all_val_accs[i]:.4f}, test {all_test_accs[i]:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
