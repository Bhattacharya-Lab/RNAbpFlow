import numpy as np
import torch, os
from src.data import utils as du

class RNADataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg, dirpath, pdbID, input_dir):
        self._samples_cfg = samples_cfg
        self.pdbID = pdbID
        self.dirpath = dirpath
        self.input_dir = input_dir

        mapfeat = du.read_pkl(f"{self.dirpath}/{self.pdbID}/map.pkl")

        onehot = mapfeat["onehot"]

        L = onehot.shape[0]

        ss_arr = self.load_bp_maps(f"{self.input_dir}/{self.pdbID}", L)
        
        all_sample_ids = []

        start_seed = 42
        
        for sample_id in range(self._samples_cfg.samples_per_sequence):
            
            seedval = start_seed * (sample_id + 1) * 10
            all_sample_ids.append((sample_id, ss_arr, seedval, onehot, mapfeat))
        
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        sample_id, ss, seedval, oh, mapfeat = self._all_sample_ids[idx]
        batch = {
            'sample_id': sample_id,
            'ss': ss,
            'seed': seedval,
            'onehot': oh,
            'mapfeat': mapfeat,

        }
        return batch

    def load_bp_maps(self, directory, seqlen):
        
        files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        num_files = len(files)

        if num_files == 0:
            sys.exit("No base pair maps provided. Exiting...")
        elif num_files > 3:
            
            print(f"More than 3 .npy maps provided. Using only the first 3 maps listed below")
            files = files[:3]
            print(files)

        maps = [np.load(os.path.join(directory, file)) for file in files]

        for map_ in maps:
            if map_.shape != (seqlen, seqlen):
                sys.exit("Inconsistent shapes in .npy files. All maps must be LxL. L = fasta sequebce length. Exiting...")
        
        if num_files == 1:
            # Duplicate the single file 3 times
            final_map = np.stack([maps[0]] * 3, axis=-1)
        elif num_files == 2:
            final_map = np.stack([maps[0], maps[1], maps[0]], axis=-1)
        elif num_files == 3:
            final_map = np.stack(maps, axis=-1)

        return final_map