# Copyright 2022 Y.K, Kihara Lab
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

import src.data.base_constants as rc
from src.data.rigid_utils import Rotation, Rigid
from src.data.tensor_utils import (
    batched_gather,
)

def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    g1_frame_to_bb = all_frames[..., 1]
    g2_frame_to_frame = all_frames[..., 2]
    g2_frame_to_bb = g1_frame_to_bb.compose(g2_frame_to_frame)
    g3_frame_to_frame = all_frames[..., 3]
    g3_frame_to_bb = g2_frame_to_bb.compose(g3_frame_to_frame)
    g4_frame_to_frame = all_frames[..., 4]
    g4_frame_to_bb = g3_frame_to_bb.compose(g4_frame_to_frame)
    g5_frame_to_frame = all_frames[..., 5]
    g5_frame_to_bb = g4_frame_to_bb.compose(g5_frame_to_frame)
    g6_frame_to_bb = all_frames[..., 6]
    g7_frame_to_bb = all_frames[..., 7]
    g8_frame_to_frame = all_frames[..., 8]
    g8_frame_to_bb = g7_frame_to_bb.compose(g8_frame_to_frame)
    g9_frame_to_bb = all_frames[..., 9]

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :1],
            g1_frame_to_bb.unsqueeze(-1),
            g2_frame_to_bb.unsqueeze(-1),
            g3_frame_to_bb.unsqueeze(-1),
            g4_frame_to_bb.unsqueeze(-1),
            g5_frame_to_bb.unsqueeze(-1),
            g6_frame_to_bb.unsqueeze(-1),
            g7_frame_to_bb.unsqueeze(-1),
            g8_frame_to_bb.unsqueeze(-1),
            g9_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom23_pos(  # was 14
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 23, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 23]
    group_mask = group_idx[aatype, ...]

    # [*, N, 23, 8]
    group_mask = nn.functional.one_hot(
        group_mask.long(),  # somehow
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 23, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 23]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 23, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 23, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions