import numpy as np, os
from pytorch_lightning import LightningModule
from src.models.flow_model import FlowModel
from src.data.interpolant_inf import Interpolant 
from src.data.nucleicacid import from_prediction, to_pdb

class FlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        
        self.model = FlowModel(cfg.model)
        self.interpolant = Interpolant(cfg.interpolant)

    def predict_step(self, batch, device):
        
        self.interpolant.set_device(device)

        sample_id = batch['sample_id'].item()
        sample_dir = self._output_dir
        
        ss = batch['ss']
        ss = ss.float()

        onehot = batch['onehot']
        onehot = onehot.float()

        seedval = batch['seed']

        map_dict = batch['mapfeat']
        
        sample_length = batch['onehot'].shape[-2]

        sample = self.interpolant.sample(1, sample_length, ss, self.model, seedval, onehot, map_dict, self._interpolant_cfg.sampling.num_timesteps)
        
        results = {}
        results['final_atom_positions'] = sample
        
        indexlist = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
        ]

        feats = {}

        feats['aatype'] = map_dict['aatype'].cpu().numpy()
        
        idx = np.arange(sample_length)
        idx = idx[None,:]
        feats['residue_index'] = idx

        typelist = list(feats['aatype'][0])

        final_atom_mask = []

        for i in range(len(typelist)):
            nttype = typelist[i]
            final_atom_mask.append(indexlist[nttype])
        
        final_atom_mask = np.array(final_atom_mask)

        results['final_atom_mask'] = np.expand_dims(final_atom_mask, axis=0)

        structure = from_prediction(feats, results)

        pdbpath = os.path.join(sample_dir, f'Sample_{sample_id}.pdb')

        with open(f"{pdbpath}", "wt") as f:
            print(to_pdb(structure), file=f)
