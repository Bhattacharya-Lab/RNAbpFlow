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

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.data.base_constants import (
    restype_rigid_group_default_frame,
    restype_atom23_to_rigid_group,
    restype_atom23_mask,
    restype_atom23_rigid_group_positions,
)

from src.data.feats import (
    frames_and_literature_positions_to_atom23_pos,
    torsion_angles_to_frames,
)

from src.data.rigid_utils import Rotation, Rigid


def to_atom23_rna(trans, rots, aatype, torsions=None):

    backb_to_global = Rigid(
        Rotation(
            rot_mats=rots,
            quats=None
        ),
        trans,
    )

    all_frames_to_global = bp_torsion_angles_to_frames(
        backb_to_global,
        torsions,
        aatype,
    )

    pred_xyz = bp_frames_and_literature_positions_to_atom23_pos(
        all_frames_to_global,
        aatype,
    )

    return pred_xyz, all_frames_to_global.to_tensor_4x4()

def _init_residue_constants(float_dtype, device):
    
    default_frames = torch.tensor(
        restype_rigid_group_default_frame,
        dtype=float_dtype,
        device=device,
        requires_grad=False,
    )
    
    group_idx = torch.tensor(
        restype_atom23_to_rigid_group,
        device=device,
        requires_grad=False,
    )
    
    atom_mask = torch.tensor(
        restype_atom23_mask,
        dtype=float_dtype,
        device=device,
        requires_grad=False,
    )
    
    lit_positions = torch.tensor(
        restype_atom23_rigid_group_positions,
        dtype=float_dtype,
        device=device,
        requires_grad=False,
    )

    return default_frames, group_idx, atom_mask, lit_positions

def _init_residue_constants2(float_dtype, device):
    
    default_frames = torch.tensor(
        restype_rigid_group_default_frame,
        dtype=float_dtype,
        device=device,
        requires_grad=False,
    )

    return default_frames

def bp_torsion_angles_to_frames(r, alpha, f):
    
    default_frames = _init_residue_constants2(alpha.dtype, alpha.device)
    
    return torsion_angles_to_frames(r, alpha, f, default_frames)

def bp_frames_and_literature_positions_to_atom23_pos(
    r, f  # [*, N, 8]  # [*, N]
):
    default_frames, group_idx, atom_mask, lit_positions = _init_residue_constants(r.get_rots().dtype, r.get_rots().device)
    
    return frames_and_literature_positions_to_atom23_pos(
        r,
        f,
        default_frames,
        group_idx,
        atom_mask,
        lit_positions,
    )
