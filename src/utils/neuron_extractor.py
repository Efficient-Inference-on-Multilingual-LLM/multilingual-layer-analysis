import os
import dotenv
dotenv.load_dotenv()
import torch
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class ActivationDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        try:
            fname = os.path.basename(fpath)
            # Parse filename format: layer_mlp-gate-up-proj_12.pt
            layer_id = int(fname.split('_')[-1].replace('.pt', ''))

            # Load to CPU RAM (fast due to 192GB available)
            t = torch.load(fpath, map_location='cpu')
            if t.dim() > 1: t = t.view(-1)
            
            # Optimization: Convert to uint8 (1 byte) to save bandwidth
            return (t > 0).to(torch.uint8), layer_id
        except Exception:
            return torch.tensor([]), -1

def collate_fn(batch):
    batch = [item for item in batch if item[1] != -1]
    if not batch: return None, None
    masks, ids = zip(*batch)
    return torch.stack(masks), torch.tensor(ids, dtype=torch.long)

class OfflineNeuronExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.model.name.split('/')[-1]
        # Update this path if needed
        self.base_dir = f"{os.getenv('EXTRACTION_RESULT_DIR')}/outputs_flores_plus/next_token/dev" 
        
        self.device = torch.device('cuda')
        print(f"🚀 Running on g2-standard-48 (L4 GPU)")
        
        # Model Metadata
        model_name_lower = self.model_name.lower()
        self.is_swiglu = any(x in model_name_lower for x in ["smollm", "llama", "mistral", "qwen"])
        
        self.num_layers = None
        self.inter_size = None

    def _infer_dimensions(self, sample_file_path):
        try:
            # 1. Load sample to get Intermediate Size
            tensor = torch.load(sample_file_path, map_location='cpu')
            if tensor.dim() > 1: tensor = tensor.view(-1)
            self.inter_size = tensor.shape[0]
            
            # 2. Determine Pattern based on the sample file found
            # (We use the sample file's own name to guess the pattern for the rest)
            fname = os.path.basename(sample_file_path)
            if "mlp-output" in fname:
                pattern = "layer_mlp-output_*.pt"
            else:
                pattern = "layer_mlp-gate-up-proj_*.pt"

            # 3. Count layers in the directory
            instance_dir = os.path.dirname(sample_file_path)
            layer_files = glob.glob(os.path.join(instance_dir, pattern))
            
            # SAFETY: If directory is incomplete, fallback to max index found + 1
            # Extract numbers from filenames to be sure
            indices = []
            for f in layer_files:
                try:
                    # Extract number between last underscore and .pt
                    idx = int(os.path.basename(f).split('_')[-1].replace('.pt', ''))
                    indices.append(idx)
                except:
                    pass
            
            if indices:
                self.num_layers = max(indices) + 1
            else:
                # Fallback if parsing fails (unlikely)
                self.num_layers = len(layer_files)

            print(f"✅ Dimensions Inferred: Layers={self.num_layers}, Inter_Size={self.inter_size}")
            
            if self.num_layers == 0:
                 raise ValueError("Detected 0 layers! Check your file pattern.")

        except Exception as e:
            raise ValueError(f"Could not infer dimensions: {e}")
    def aggregate_language(self, language_code, mode="raw"):
        # 1. Glob Files
        if "gpt-oss" in self.model_name.lower() or self.cfg.processing.component_type == "mlp_out":
            file_pattern = "layer_mlp-output_*.pt"
        else:
            file_pattern = "layer_mlp-gate-up-proj_*.pt"
        search_pattern = os.path.join(
            self.base_dir, self.model_name, mode, language_code, 
            "*", "last_token", file_pattern
        )
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"⚠️ No files found for {language_code} in {mode}")
            return None, 0

        if self.num_layers is None:
            self._infer_dimensions(files[0])

        cpu_count = os.cpu_count() or 4
        batch_size = 2048
        workers = min(8, cpu_count - 1)
        dataset = ActivationDataset(files)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,     
            shuffle=False, 
            num_workers=workers,     
            collate_fn=collate_fn,
            pin_memory=True,     
            prefetch_factor=2,   
            persistent_workers=True 
        )

        # 3. Initialize Counter on GPU
        total_counter = torch.zeros((self.num_layers, self.inter_size), dtype=torch.int32, device=self.device)
        
        print(f"⚡ Processing {len(files)} files with {workers} workers (Batch={batch_size})...")

        # 4. Aggregation Loop
        for batch_masks, batch_ids in tqdm(loader, leave=False, desc=f"Aggregating {language_code}"):
            if batch_masks is None: continue
            
            # Non-blocking transfer to L4 GPU
            batch_masks = batch_masks.to(self.device, non_blocking=True)
            batch_ids = batch_ids.to(self.device, non_blocking=True)
            
            # Vectorized addition
            total_counter.index_add_(0, batch_ids, batch_masks.to(torch.int32))

        n_samples = len(files) // self.num_layers if self.num_layers > 0 else 0
        return total_counter.cpu(), n_samples