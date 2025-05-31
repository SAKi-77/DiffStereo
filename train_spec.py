import os
import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusion import create_diffusion
from models import DiT
from datasets import get_dataset
from tools.data_processing import data_processing
from tools.transforms_audio.audio2spec import AudioToSpec
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard



def train(args):

    # Load hyperparameters from YAML file
    config_yaml = args.config_yaml
    with open(config_yaml, 'r') as file:
        config = yaml.safe_load(file)

    # Dataset & DataLoader
    train_dataset = get_dataset(cfg=config["data"])  # E.g., MUSDB18HQ dataset
    dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Model and optimizer
    device = config['device'] if torch.cuda.is_available() else "cpu"
    model = DiT(
        input_size=tuple(config['input_size']),
        patch_size=config['patch_size'],
        in_channels=config['in_channels'], 
        hidden_size=config['hidden_size'],
        depth=config['depth'],
        num_heads=config['num_heads'],
    )
    model.to(device)
     
    # Create diffusion model
    diffusion = create_diffusion(timestep_respacing="")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    # Create directory to save model checkpoints
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Training Loop
    num_epochs=config['num_epochs']
    sample_interval=config['sample_interval']
    
    model.train()
    epoch = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # Preprocess input and condition
            x, cond_tf = data_processing(
                data=data,
                preprocess_func=AudioToSpec().to(device),
                device=device
            ) # x:(b, c*2, t, f), cond_tf:(b, c, t, f)
            
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
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/epoch', avg_loss, epoch + 1)

        if (epoch + 1) % sample_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_path = f"{config['checkpoint_dir']}/model_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    args = parser.parse_args()
    
    writer = SummaryWriter()
    
    train(args)
    
    writer.close()

