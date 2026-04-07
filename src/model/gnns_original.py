from __future__ import annotations
import argparse
import os
import math
import numpy as np
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ==================== Utility Functions (from layers.py) ====================

def add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Add self-loops to edge_index.
    edge_index: LongTensor [2, E]
    returns: LongTensor [2, E + num_nodes]
    """
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    self_loops = torch.stack([self_loops, self_loops], dim=0)  # [2, N]
    return torch.cat([edge_index, self_loops], dim=1)


def degree(index: Tensor, num_nodes: int, device=None, dtype=torch.float32) -> Tensor:
    """
    Compute (in-)degree for nodes given edge target indices.
    index: LongTensor [E] (dst indices)
    returns: Tensor [N]
    """
    if device is None:
        device = index.device
    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    one = torch.ones(index.size(0), device=device, dtype=dtype)
    deg.index_add_(0, index, one)
    return deg


# ==================== Graph Neural Network Layers (from layers.py) ====================

class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer (Kipf & Welling, 2017)
    Uses symmetric normalization: D^{-1/2} (A + I) D^{-1/2}
    edge_index is treated as directed edges; for best results, provide undirected edges (both directions).
    """

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
        deg_dst = degree(dst, num_nodes, device=edge_index.device)  # in-degree        
        deg_src = degree(src, num_nodes, device=edge_index.device)  # out-degree
        deg_inv_sqrt_dst = (deg_dst + 1e-12).pow(-0.5)
        deg_inv_sqrt_src = (deg_src + 1e-12).pow(-0.5)
        norm = deg_inv_sqrt_src[src] * deg_inv_sqrt_dst[dst]  # [E]
        return norm

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        if num_nodes is None:
            num_nodes = x.size(0)
        if self.add_self_loops_flag:
            edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index
        xW = self.lin(x)  # [N, Fout]
        norm = self._normalized_edge_weight(edge_index, num_nodes)  # [E]

        out = torch.zeros(num_nodes, xW.size(1), device=x.device, dtype=xW.dtype)
        messages = xW[src] * norm.unsqueeze(-1)  # [E, Fout]
        out.index_add_(0, dst, messages)  # aggregate to dst

        if self.bias is not None:
            out = out + self.bias
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Layer (Velickovic et al., 2018)
    Multi-head attention; for hidden layers we usually concat heads; for final layer we average heads.
    Softmax over incoming neighbors per destination node.
    """

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
        assert heads >= 1
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

        src, dst = edge_index  # [E], [E]

        h = self.lin(x)  # [N, H*out]
        h = h.view(-1, self.heads, self.out_channels)  # [N, H, C]

        h_src = h[src]  # [E, H, C]
        h_dst = h[dst]  # [E, H, C]

        e_src = (h_src * self.att_src).sum(dim=-1)  # [E, H]
        e_dst = (h_dst * self.att_dst).sum(dim=-1)  # [E, H]
        e = F.leaky_relu(e_src + e_dst, negative_slope=self.negative_slope)  # [E, H]

        exp_e = torch.exp(e)
        denom = torch.zeros(num_nodes, self.heads, device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, exp_e)  # [N, H]
        alpha = exp_e / (denom[dst] + 1e-16)  # [E, H]

        if self.training and self.dropout > 0.0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        messages = h_src * alpha.unsqueeze(-1)  # [E, H, C]
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, messages)  # [N, H, C]

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_channels)  # [N, H*C]
        else:
            out = out.mean(dim=1)  # [N, C]

        if self.bias is not None:
            out = out + self.bias
        return out


class SAGEConv(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017) - Mean aggregator.
    """

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

        src, dst = edge_index  # [E], [E]

        neigh_sum = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        neigh_sum.index_add_(0, dst, x[src])

        deg = degree(dst, num_nodes, device=x.device, dtype=x.dtype).clamp(min=1.0).unsqueeze(-1)  # [N, 1]
        neigh_mean = neigh_sum / deg  # [N, Fin]

        out = self.lin_neigh(neigh_mean) + self.lin_self(x)  # [N, Fout]
        return out


# ==================== Full Models (from models.py) ====================

class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        add_self_loops: bool = True,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 2, "GCN requires at least 2 layers."
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        self.layers.append(GCNLayer(in_channels, hidden_channels, add_self_loops_flag=add_self_loops))
        if self.bns is not None:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_channels, hidden_channels, add_self_loops_flag=add_self_loops))
            if self.bns is not None:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.layers.append(GCNLayer(hidden_channels, out_channels, add_self_loops_flag=add_self_loops))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        out_heads: int = 1,
        dropout: float = 0.6,
        attn_dropout: float = 0.6,
        add_self_loops: bool = True,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 2, "GAT requires at least 2 layers."
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

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

        for _ in range(num_layers - 2):
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

        self.layers.append(
            GATLayer(
                hidden_channels * heads,
                out_channels,
                heads=out_heads,
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


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 2, "GraphSAGE requires at least 2 layers."
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        self.layers.append(SAGEConv(in_channels, hidden_channels))
        if self.bns is not None:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
            if self.bns is not None:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ==================== Training Utilities (from train_npz.py) ====================

def parse_args():
    p = argparse.ArgumentParser(description="Train GCN/GAT/GraphSAGE on .npz graph data")
    p.add_argument("--data", type=str, required=True, help="Path to dataset .npz file")
    p.add_argument("--model", type=str, default="sage", choices=["gcn", "gat", "sage"], help="Model type")
    p.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    p.add_argument("--layers", type=int, default=2, help="Number of layers (>=2)")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    p.add_argument("--heads", type=int, default=8, help="GAT heads for hidden layers")
    p.add_argument("--out-heads", type=int, default=1, help="GAT heads for output layer (averaged)")
    p.add_argument("--attn-dropout", type=float, default=0.6, help="GAT attention dropout")
    p.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--seeds", nargs='+', type=int, default=[42,420,403,86,860,71,55,128,159,223], help="Random seeds (default: [0, 5, 565, 3265])")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    p.add_argument("--no-make-undirected", action="store_false", dest="make_undirected", default=True,help="Disable making graph undirected")
    p.add_argument("--transpose-edges", action="store_true", help="If edges are shape [E,2], transpose to [2,E]")
    p.add_argument("--no-self-loops", action="store_true", help="Disable adding self-loops inside layers")
    p.add_argument("--use-bn", action="store_true", help="Use BatchNorm between layers")
    p.add_argument("--checkpoint", type=str, default="", help="Path to save best model state_dict")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    # Concatenate reversed edges; does not deduplicate (not necessary for these ops)
    #return torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
    return edge_index_undirected


def load_npz(path: str, device: str = "cpu", transpose_edges: bool = False, make_undirected: bool = False):
    data = np.load(path, allow_pickle=True)
    # Required fields
    edges = data["edges"]
    node_features = data["node_features"]
    node_labels = data["node_labels"]
    train_masks = data["train_masks"]
    val_masks = data["val_masks"]
    test_masks = data["test_masks"]

    # Convert to torch
    if transpose_edges or (edges.ndim == 2 and edges.shape[0] != 2 and edges.shape[1] == 2):
        edges = edges.T  # to [2, E]
    assert edges.shape[0] == 2, f"edges must be shape [2, E], got {edges.shape}"
    edge_index = torch.from_numpy(edges.astype(np.int64, copy=False))
    x = torch.from_numpy(node_features.astype(np.float32, copy=False))
    y = torch.from_numpy(node_labels.astype(np.int64, copy=False))
    train_mask = torch.from_numpy(train_masks.astype(bool, copy=False))
    val_mask = torch.from_numpy(val_masks.astype(bool, copy=False))
    test_mask = torch.from_numpy(test_masks.astype(bool, copy=False))

    if make_undirected:
        edge_index = to_undirected(edge_index)

    # Move to device
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    return x, edge_index, y, train_mask, val_mask, test_mask


def build_model(
    model_name: str,
    in_dim: int,
    num_classes: int,
    hidden: int,
    layers: int,
    dropout: float,
    heads: int,
    out_heads: int,
    attn_dropout: float,
    add_self_loops: bool,
    use_bn: bool,
):
    if model_name == "gcn":
        model = GCN(
            in_channels=in_dim,
            hidden_channels=hidden,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
            add_self_loops=add_self_loops,
            use_batchnorm=use_bn,
        )
    elif model_name == "gat":
        model = GAT(
            in_channels=in_dim,
            hidden_channels=hidden,
            out_channels=num_classes,
            num_layers=layers,
            heads=heads,
            out_heads=out_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            add_self_loops=add_self_loops,
            use_batchnorm=use_bn,
        )
    else:
        model = GraphSAGE(
            in_channels=in_dim,
            hidden_channels=hidden,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
            use_batchnorm=use_bn,
        )
    return model


def train_one_epoch(model, x, edge_index, y, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)  # [N, C]
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_metrics(model, x, edge_index, y, mask):
    model.eval()
    logits = model(x, edge_index)
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == y[mask]).sum().item()
    total = int(mask.sum().item())
    acc = correct / max(total, 1)
    return acc, logits


def run_single_seed(seed, args, x, edge_index, y, train_mask, val_mask, test_mask, in_dim, num_classes, device):
    """Run the full experiment for a single seed"""
    print(f"\n========== Running with seed {seed} ==========")
    set_seed(seed)
    
    model = build_model(
        args.model,
        in_dim,
        num_classes,
        args.hidden,
        args.layers,
        args.dropout,
        args.heads,
        args.out_heads,
        args.attn_dropout,
        add_self_loops=not args.no_self_loops,
        use_bn=args.use_bn,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = 0.0
    best_test_at_val = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, x, edge_index, y, train_mask, optimizer)
        val_acc, _ = eval_metrics(model, x, edge_index, y, val_mask)
        test_acc, _ = eval_metrics(model, x, edge_index, y, test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test_at_val = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | val {val_acc:.4f} | "
                f"test {test_acc:.4f} | best_val {best_val:.4f} | best_test_at_val {best_test_at_val:.4f}"
            )

    print(f"Seed {seed} finished. Best val acc: {best_val:.4f}, Test acc at best val: {best_test_at_val:.4f}")
    
    # Save model (if a checkpoint path is specified)
    if args.checkpoint and best_state is not None:
        checkpoint_path = args.checkpoint.replace('.pth', f'_seed{seed}.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_state, checkpoint_path)
        print(f"Saved best model to {checkpoint_path}")
    
    return best_val, best_test_at_val


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load data
    x, edge_index, y, train_mask, val_mask, test_mask = load_npz(
        args.data, device=device, transpose_edges=args.transpose_edges, make_undirected=args.make_undirected
    )

    in_dim = int(x.size(1))
    num_classes = int(y.max().item() + 1)

    print(f"Dataset loaded: {x.size(0)} nodes, {edge_index.size(1)} edges, {in_dim} features, {num_classes} classes")
    print(f"Running experiments with seeds: {args.seeds}")

    # Run experiments across multiple seeds
    all_val_accs = []
    all_test_accs = []
    
    for seed in args.seeds:
        val_acc, test_acc = run_single_seed(
            seed, args, x, edge_index, y, train_mask, val_mask, test_mask, 
            in_dim, num_classes, device
        )
        all_val_accs.append(val_acc)
        all_test_accs.append(test_acc)
    
    # Print summary statistics
    print(f"\n========== Summary across all seeds ==========")
    print(f"Val accuracy: {np.mean(all_val_accs):.4f} ± {np.std(all_val_accs):.4f}")
    print(f"Test accuracy: {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}")
    print(f"Individual results:")
    for i, seed in enumerate(args.seeds):
        print(f"  Seed {seed}: val {all_val_accs[i]:.4f}, test {all_test_accs[i]:.4f}")


if __name__ == "__main__":
    main()
