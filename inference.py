import os, sys
import time
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import src.utils as eu
from src.models.flow_module_inf import FlowModule
from src.data.dataset import RNADataset

import pickle, yaml, shutil
from tqdm import tqdm
from src.data.data_transform import make_atom_mask
import torch.multiprocessing as mp

torch.set_float32_matmul_precision('high')

class Sampler:
    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        
        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._interpolant_cfg = self._infer_cfg.interpolant
        self._target = self._infer_cfg.name
        self._input_dir = self._infer_cfg.input_dir
        
        # Read checkpoint and initialize module.
        print()
        print(f"==++++Model loaded for inference = {ckpt_path}====++++")
        print()
        
        # Dynamically get available GPUs
        gpu_count = torch.cuda.device_count()

        if gpu_count > 0:
            device_ids = [f"cuda:{i}" for i in range(gpu_count)]
            map_location = lambda storage, loc: storage.cuda(device_ids[0])  # Load to the first GPU
        else:
            map_location = "cpu"  # Fallback to CPU
            
        self._flow_module = FlowModule.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
        
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._interpolant_cfg = self._interpolant_cfg
        self._flow_module._output_dir = os.path.join(self._infer_cfg.output_dir, self._target)

        self.batch_list = []

    def send_to_device(self, data, device):
        if isinstance(data, torch.Tensor):  # If it's a tensor, move to device
            return data.to(device)
        elif isinstance(data, dict):  # If it's a dictionary, recurse on values
            return {k: self.send_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):  # If it's a list, recurse on elements
            return [self.send_to_device(v, device) for v in data]
        elif isinstance(data, tuple):  # If it's a tuple, recurse and return a tuple
            return tuple(self.send_to_device(v, device) for v in data)
        # Return as-is for unsupported types
        return data
    
    def sample_GPU(self, process_index):
        # Cycle through GPUs using modulo
        gpu_index = process_index % self._infer_cfg.num_gpus
        device = torch.device(f"cuda:{gpu_index}")  # Assign device based on GPU index

        model = self._flow_module.to(device)
        
        # Calculate the batches for this process
        assigned_batches = [i for i in range(process_index, len(self.batch_list), self._infer_cfg.num_gpus)]
        
        for batch_index in tqdm(range(len(assigned_batches))):
            batch = self.batch_list[assigned_batches[batch_index]]  # Get the corresponding batch
            
            batch = self.send_to_device(batch, device)
            with torch.no_grad():
                model.predict_step(batch, device)

    def run_sampling(self):
    
        eval_dataset = RNADataset(self._samples_cfg, self._infer_cfg.output_dir, self._target, self._input_dir)
        
        dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

        for batch in dataloader:
            self.batch_list.append(batch)
        
        start_time = time.time()
        
        gpu_count = torch.cuda.device_count()

        if gpu_count < self._infer_cfg.num_gpus:
            num_procs = gpu_count
        else:
            num_procs = self._infer_cfg.num_gpus

        if len(self.batch_list) < num_procs:
            num_procs = len(self.batch_list)

        print(f'Starting inference with {num_procs} GPUs:')

        mp.spawn(self.sample_GPU, nprocs=num_procs, join=True)

        elapsed_time = time.time() - start_time
        print(f'Finished in {elapsed_time:.2f}s')
        

@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def run(cfg: DictConfig) -> None:

    sampler = Sampler(cfg)
    sampler.run_sampling()

def run_inference():

    CONFIG_FILE_PATH = "configs/inference.yaml"

    with open(CONFIG_FILE_PATH, 'r') as file:
        yaml_content = yaml.safe_load(file)

    input_dir = yaml_content['inference']['input_dir']
    output_dir = yaml_content['inference']['output_dir']

    list_file_path = os.path.join(input_dir, "list.txt")
    with open(list_file_path, "r") as file:
        lines = file.readlines()

    id_list = []
    sample_count_list = []

    for line in lines:
        tokens = line.split()
        id_list.append(tokens[0].strip())
        if len(tokens) == 2:
            sample_count_list.append(tokens[1].strip())

    for idx, target_id in enumerate(tqdm(id_list)):
        yaml_content['inference']['name'] = target_id

        if len(sample_count_list) == len(id_list):
            yaml_content['inference']['samples']['samples_per_sequence'] = int(sample_count_list[idx])

        with open(CONFIG_FILE_PATH, 'w') as file:
            yaml.dump(yaml_content, file, default_flow_style=False, sort_keys=False)

        target_dir = os.path.join(output_dir, target_id)
        os.makedirs(target_dir, exist_ok=True)

        atom_dict = make_atom_mask(target_id, input_dir)
        pickle_file_path = os.path.join(target_dir, "map.pkl")
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(atom_dict, file)

        run()

        os.remove(pickle_file_path)

if __name__ == '__main__':
    
    run_inference()

