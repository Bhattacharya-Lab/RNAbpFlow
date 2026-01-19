import os
import time
import hydra
import torch
import pickle
import yaml
import torch.multiprocessing as mp

from tqdm import tqdm
from omegaconf import DictConfig
from src.models.flow_module_inf import FlowModule
from src.data.dataset import RNADataset
from src.data.data_transform import make_atom_mask

torch.set_float32_matmul_precision('high')

class Sampler:
    def __init__(self, cfg: DictConfig):
        """Initialize the sampler and model once."""
        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._interpolant_cfg = self._infer_cfg.interpolant
        self._input_dir = self._infer_cfg.input_dir
        self._output_dir_base = self._infer_cfg.output_dir

        ckpt_path = self._infer_cfg.ckpt_path
        print(f"\n==++++Model loaded for inference = {ckpt_path}====++++\n")

        gpu_count = torch.cuda.device_count()
        map_location = (lambda storage, loc: storage.cuda(0)) if gpu_count > 0 else "cpu"

        self._flow_module = FlowModule.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._interpolant_cfg = self._interpolant_cfg
        self.batch_list = []

    def send_to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self.send_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.send_to_device(v, device) for v in data]
        elif isinstance(data, tuple):
            return tuple(self.send_to_device(v, device) for v in data)
        return data

    def sample_GPU(self, process_index, world_size):
        gpu_index = process_index % world_size
        device = torch.device(f"cuda:{gpu_index}")
        model = self._flow_module.to(device)

        assigned_batches = [i for i in range(process_index, len(self.batch_list), world_size)]

        for batch_index in tqdm(range(len(assigned_batches))):
            batch = self.batch_list[assigned_batches[batch_index]]
            batch = self.send_to_device(batch, device)
            with torch.no_grad():
                model.predict_step(batch, device)
    
    def run_sampling(self, target_id):
        self._infer_cfg.name = target_id
        self._flow_module._output_dir = os.path.join(self._output_dir_base, target_id)

        eval_dataset = RNADataset(self._samples_cfg, self._output_dir_base, target_id, self._input_dir)
        dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

        self.batch_list = [batch for batch in dataloader]

        gpu_count = torch.cuda.device_count()
        num_procs = min(self._infer_cfg.num_gpus, gpu_count, len(self.batch_list))

        print(f"Starting inference on target {target_id} with {num_procs} GPUs...")
        start_time = time.time()

        mp.spawn(self.sample_GPU, args=(num_procs,), nprocs=num_procs, join=True)
        
        elapsed_time = time.time() - start_time
        print(f"Finished {target_id} in {elapsed_time:.2f}s")


def run_inference():
    CONFIG_FILE_PATH = "configs/inference.yaml"

    with open(CONFIG_FILE_PATH, 'r') as file:
        yaml_content = yaml.safe_load(file)

    input_dir = yaml_content['inference']['input_dir']
    output_dir = yaml_content['inference']['output_dir']
    cfg = DictConfig(yaml_content)

    list_file_path = os.path.join(input_dir, "list.txt")
    with open(list_file_path, "r") as file:
        lines = file.readlines()

    id_list = []
    sample_count_list = []
    for line in lines:
        tokens = line.strip().split()
        id_list.append(tokens[0])
        if len(tokens) == 2:
            sample_count_list.append(int(tokens[1]))

    sampler = Sampler(cfg)

    for idx, target_id in enumerate(tqdm(id_list)):
        if len(sample_count_list) == len(id_list):
            sampler._samples_cfg.samples_per_sequence = sample_count_list[idx]

        target_dir = os.path.join(output_dir, target_id)
        os.makedirs(target_dir, exist_ok=True)

        atom_dict = make_atom_mask(target_id, input_dir)
        pickle_file_path = os.path.join(target_dir, "map.pkl")
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(atom_dict, f)

        sampler.run_sampling(target_id)

        os.remove(pickle_file_path)


if __name__ == '__main__':
    run_inference()
