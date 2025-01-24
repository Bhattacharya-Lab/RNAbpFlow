"""
Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/models/edge_embedder.py
"""

import torch
from torch import nn
from src.models import utils

class EdgeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 1 + 3
        #total_edge_feats = self.feat_dim * 3 + 3

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, pos):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = utils.get_index_embedding(rel_pos, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, p_mask, ss):
        num_batch, num_res, _ = s.shape

        p_i = self.linear_s_p(s) #Converts the node feature Lx256 to Lx64

        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)
        
        pos = torch.arange(
            num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        
        relpos_feats = self.embed_relpos(pos)
        
        sc_feats = utils.calc_distogram(
             sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        
        all_edge_feats = torch.concat(
           [cross_node_feats, relpos_feats, sc_feats, ss], dim=-1)
        
        edge_feats = self.edge_embedder(all_edge_feats)
        edge_feats *= p_mask.unsqueeze(-1)
        
        return edge_feats