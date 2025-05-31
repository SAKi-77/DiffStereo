import os
import argparse
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from diffusion import create_diffusion
from models import DiT
from datasets import get_dataset
from tools.data_processing import data_processing
from tools.transforms_audio.audio2spec import AudioToSpec
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
# use DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


local_rank = int(os.getenv('LOCAL_RANK', -1))

parser = argparse.ArgumentParser()
parser.add_argument('--config_yaml', default="/home/suqi/Mono2Stereo/config/train.yaml", type=str, required=True)
parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
args = parser.parse_args()

# Load hyperparameters from YAML file
config_yaml = args.config_yaml
with open(config_yaml, 'r') as file:
    config = yaml.safe_load(file)


# DDP: Create TensorBoard writer (only for rank 0)
if local_rank == 0:
    writer = SummaryWriter()


# DDP: backend initialization
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')


# Dataset & DataLoader
train_dataset = get_dataset(cfg=config["data"])  # E.g., MUSDB18HQ dataset
train_sampler = DistributedSampler(train_dataset)
dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler)

# Model and optimizer
device = torch.device("cuda", local_rank)
model = DiT(
    input_size=tuple(config['input_size']),
    patch_size=config['patch_size'],
    in_channels=config['in_channels'], 
    hidden_size=config['hidden_size'],
    depth=config['depth'],
    num_heads=config['num_heads'],
)
model.to(device)

# DDP: Load model
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Create diffusion model
diffusion = create_diffusion(timestep_respacing="")

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

# Create directory to save model checkpoints
os.makedirs(config['checkpoint_dir'], exist_ok=True)

# Training Loop
num_epochs=config['num_epochs']
sample_interval=config['sample_interval']
def train_model(model, dataloader, optimizer, diffusion, num_epochs, sample_interval):
    model.train()
    epoch = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        dataloader.sampler.set_epoch(epoch) # DDP: Set epoch for DistributedSampler
        for step, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # Preprocess input and condition
            x, cond_tf = data_processing(
                data=data,
                preprocess_func=AudioToSpec().to(device),
                device=device
            ) # x, cond_tf: (b, c, t, f)
            
            # Sample t
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=cond_tf)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Only log and save checkpoints for rank 0
        if dist.get_rank() == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss: {avg_loss:.4f}")
            writer.add_scalar('Loss/epoch', avg_loss, epoch + 1)

            if (epoch + 1) % sample_interval == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                checkpoint_path = f"{config['checkpoint_dir']}/model_epoch_{epoch + 1}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Model checkpoint saved at epoch {epoch + 1}")
        

# Start training
train_model(model, dataloader, optimizer, diffusion, num_epochs=config['num_epochs'], sample_interval=config['sample_interval'])

# Close TensorBoard writer
if dist.get_rank() == 0:
    writer.close()

# Clean up the process group
dist.destroy_process_group()
    

################
# CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.launch --nproc_per_node 2 --master_port=29602 train_spec_ddp.py 
