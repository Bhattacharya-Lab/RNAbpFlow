#Adapted from Fraemflow
#Credit: https://github.com/microsoft/protein-frame-flow

import torch, copy
from src.data import so3_utils
from src.data import utils as du
from src.data import all_atom as rna_all_atom
import numpy as np

from src.data.tensor_utils import batched_gather

from scipy.spatial.transform import Rotation

def _centered_gaussian(num_batch, num_res, device, seedval):
    # returns randomly-sampled zero-centered Gaussian noise for translations
    if seedval != -1:
        torch.manual_seed(seedval)
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device, seedval):
    # returns uniformly-sampled rotations (from U(SO(3)))
    if seedval != -1:
        np.random.seed(seedval)
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(), # random rotation
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

class Interpolant:
    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    def set_device(self, device):
        self._device = device

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        """
        Perform Euler ODESolve step for translations
        """
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        """
        Perform Euler ODESolve step for rotations based on prescribed sampling schedule
        """
        scaling = 10
        
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

    def sample(self, num_batch, num_res, ss, model, seedval, onehot, map_dict, num_timesteps):
        """
        Params:
            num_batch : number of independent samples
            num_res : number of nucleotides to model
            model : the parameterised vector field (ie, the "denoiser")

        Returns:
            Generated full atom RNA 3D structures
        """

        res_mask = torch.ones(num_batch, num_res, device=self._device)
        
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(num_batch, num_res, self._device, seedval) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device, seedval)
        
        batch = {
            'res_mask': res_mask,
        }

        # get diffusion timesteps in order between [0, 1]
        ts = torch.linspace(1e-2, 1.0, num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]

        prev_trans = trans_0
        prev_rot = rotmats_0

        for t_2 in ts[1:]:
            # run one step of reverse diffusion (denoising)
            trans_t_1, rotmats_t_1 = prev_trans, prev_rot

            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            batch['ss'] = ss

            batch['onehot'] = onehot
            
            with torch.no_grad():
                model_out = model(batch)

            # store model outputs
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            
            batch['trans_sc'] = pred_trans_1

            # take reverse step
            d_t = t_2 - t_1
            
            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            
            prev_trans = trans_t_2
            prev_rot = rotmats_t_2

            t_1 = t_2

        # NOTE: we only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prev_trans, prev_rot
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        
        with torch.no_grad():
            model_out = model(batch) # final predicted frames (translations + rotations)
        
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        pred_torsions_1 = model_out['pred_torsions']
        
        is_na_residue_mask = torch.ones(num_batch, num_res, device=self._device).bool()

        assert res_mask.shape == is_na_residue_mask.shape, "Shape mismatch between NA masks"

        pred_bb_atoms_23,_ = rna_all_atom.to_atom23_rna(
                            pred_trans_1, pred_rotmats_1, 
                            map_dict['aatype'],
                            torsions=pred_torsions_1,
                        )

        atom28_data = batched_gather(
            pred_bb_atoms_23,
            map_dict["atom_map"],
            dim=-2,
            no_batch_dims=len(pred_bb_atoms_23.shape[:-2]),
        )

        
        return atom28_data.detach().cpu().numpy()