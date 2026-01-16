import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict

cudaName = 'cuda:0'


class Conv1dReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class MHSA(nn.Module):
    def __init__(self, embed_dim=96, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x):
        B, L, D = x.shape 
        
        Q = self.q_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V) # [B, num_heads, L, d_k]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_linear(attn_out)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        layers = []
        for i in range(layer_num):
            ic = in_channels if i == 0 else out_channels
            layers.append(Conv1dReLU(ic, out_channels, kernel_size, stride, padding))
        self.inc = nn.Sequential(*layers)

    def forward(self, x):
        return self.inc(x) 


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class CTNBlock(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList([
            StackCNN(i + 1, embedding_num, 96, 3, padding=1) for i in range(block_num)
        ])
        
        self.linear = nn.Linear(block_num * 96, 96)
        self.MHSA = MHSA(96, 8)
        self.MLP = MLP(96, embedding_num, 96)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        
        feats = [block(x) for block in self.block_list]
        
        combined_seq = torch.cat(feats, dim=1) # [B, 96*3, L]
        combined_seq = combined_seq.permute(0, 2, 1) # [B, L, 96*3]
        x_hid = self.linear(combined_seq) # [B, L, 96]
        
        o1 = self.MHSA(x_hid) + x_hid      # Residual 1
        o3 = self.MLP(o1) + o1            # Residual 2
        
        layer1 = self.pool(feats[0]).squeeze(-1)
        layer2 = self.pool(feats[1]).squeeze(-1)
        layer3 = self.pool(feats[2]).squeeze(-1)
        
        final_x = self.pool(o3.permute(0, 2, 1)).squeeze(-1)
        
        return final_x, layer1, layer2, layer3


class NodeLevelBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)

class GDC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

        self.diffusion_iterations = 2
        self.diffusion_weight = 0.1

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv2(self.conv1(x, edge_index), edge_index)))

        adj_t = torch.sparse_coo_tensor(edge_index, torch.ones_like(edge_index[0]).float(), (x.size(0), x.size(0)))
        adj_t = adj_t.to_dense()
        adj_t = adj_t + torch.eye(x.size(0)).to(x.device)
        for t in range(self.diffusion_iterations):
            adj_t = self.diffusion_weight * torch.matmul(adj_t, adj_t) + (1 - self.diffusion_weight) * torch.eye(
                x.size(0)).to(x.device)
        adj_t = adj_t / adj_t.sum(dim=1, keepdim=True)

        laplacian = torch.eye(adj_t.size(0)).to(x.device) - adj_t
        reg_term = torch.trace(torch.matmul(torch.transpose(x, 0, 1), laplacian))

        data.x = torch.matmul(adj_t, x) + 0.01 * reg_term

        return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        mid_channels = int(growth_rate * bn_size)
        self.conv1 = GDC(num_input_features, mid_channels, mid_channels)
        self.conv2 = GDC(mid_channels, growth_rate, growth_rate)

    def forward(self, data):
        temp_data = data.clone()
        temp_data = self.conv1(temp_data)
        temp_data = self.conv2(temp_data)
        return temp_data.x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            for i in range(num_layers)
        ])

    def forward(self, data):
        for layer in self.layers:
            new_features = layer(data)
            data.x = torch.cat([data.x, new_features], dim=1)
        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GDC(num_input_features, 32, 32))]))
        num_input_features = 32
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GDC(num_input_features, num_input_features // 2, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            print('transition_num:', trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = gnn.global_mean_pool(data.x, data.batch)
        x = self.classifer(x)

        return x


class LIIIS(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(LIIIS, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, input_dim)

        self.regularization = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=False),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=False)
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        q = self.query(x1).view(batch_size, self.num_heads, 1, self.head_dim)
        k = self.key(x2).view(batch_size, self.num_heads, 1, self.head_dim)
        v = self.value(x2).view(batch_size, self.num_heads, 1, self.head_dim)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, v).view(batch_size, -1)

        out = self.fc(attention_output)
        out = self.regularization(out)

        out = out + x1

        return self.layer_norm(out)


class MDCTDTA(nn.Module):
    def __init__(self, block_num, vocab_protein_size, embedding_size=128, filter_num=32, out_dim=1):
        super(MDCTDTA, self).__init__()
        self.protein_encoder = CTNBlock(block_num, vocab_protein_size, embedding_size)

        self.ligand_encoder1 = GraphDenseNet(22, 96, block_config=[8], bn_sizes=[2])
        self.ligand_encoder2 = GraphDenseNet(144, 96, block_config=[8, 8], bn_sizes=[2, 2])
        self.ligand_encoder3 = GraphDenseNet(200, 96, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])

        self.attn_x1 = LIIIS(96, 32)
        self.attn_y1 = LIIIS(96, 32)
        self.attn_x2 = LIIIS(96, 32)
        self.attn_y2 = LIIIS(96, 32)
        self.attn_x3 = LIIIS(96, 32)
        self.attn_y3 = LIIIS(96, 32)

        self.merge = nn.Linear(96 * 3, 96)

        self.classifier = nn.Sequential(
            nn.Linear(96 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )


    def forward(self, data):
        target = data.target
        _, layer1, layer2, layer3 = self.protein_encoder(target)

        lx1 = self.ligand_encoder1(data)
        d1 = self.attn_x1(layer1, lx1)
        p1 = self.attn_y1(lx1, layer1)

        lx2 = self.ligand_encoder2(data)
        d2 = self.attn_x2(layer2, lx2)
        p2 = self.attn_y2(lx2, layer2)

        lx3 = self.ligand_encoder3(data)
        d3 = self.attn_x3(layer3, lx3)
        p3 = self.attn_y3(lx3, layer3)

        protein_combined = self.merge(torch.cat([d1, d2, d3], dim=-1))
        ligand_combined = self.merge(torch.cat([p1, p2, p3], dim=-1))

        combined_all = torch.cat([protein_combined, ligand_combined], dim=-1)
        return self.classifier(combined_all)

