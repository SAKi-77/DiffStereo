import torch
import soundfile as sf
from models import DiT
from diffusion import create_diffusion
from tqdm import tqdm
from torch.utils.data import DataLoader
from tools.transforms_audio.audio2spec import AudioToSpec
from tools.transforms_audio.spec2audio import SpecToAudio
from pathlib import Path
from audidata.datasets import MUSDB18HQ
from audidata.io.crops import RandomCrop

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_trained_model(checkpoint_path):
    model = DiT(
        input_size=(729, 513),
        patch_size=9,
        in_channels=4, 
        hidden_size=384,
        depth=12,
        num_heads=6,
    )
    model.to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

class MUSDB18HQDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='test', sr=24000, clip_duration=9.1):
        self.dataset = MUSDB18HQ(root=root, split=split, sr=sr, crop=RandomCrop(clip_duration))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        mono_condition = torch.tensor(sample["mixture"]).mean(dim=0).unsqueeze(0)  # (1, t)
        mixture = torch.tensor(sample["mixture"])  # (c, t)
        return mono_condition.to(device), mixture.to(device)

def infer_and_generate_audio(model, diffusion, spec_cond):
    latent_size = (729, 513)
    z = torch.randn(1, 4, latent_size[0], latent_size[1], device=device)  # (b, c, t, f)
    model_kwargs = dict(y=spec_cond)

    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    
    return samples

def save_audio(audio, output_path, it, sr=24000):
    audio = audio.cpu().numpy()
    output_path = Path(output_path, f"sample{it}.wav")
    if audio.shape[0] == 1:
        audio = audio.squeeze(0)  
        sf.write(file=output_path, data=audio, samplerate=sr)
    else:
        sf.write(file=output_path, data=audio.T, samplerate=sr)
    
    print(f"Write out to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate audio using DiT')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_gt_dir', type=str, required=True, help='Directory to save ground truth audio')
    parser.add_argument('--output_gt_mono_dir', type=str, required=True, help='Directory to save ground truth mono audio')
    parser.add_argument('--output_gen_dir', type=str, required=True, help='Directory to save generated audio')
    parser.add_argument('--sample_rate', type=int, default=24000, help='Sample rate for output audio')
    args = parser.parse_args()

    model = load_trained_model(args.checkpoint)
    diffusion = create_diffusion(timestep_respacing="")
    audio2spec = AudioToSpec()
    spec2audio = SpecToAudio()

    test_dataset = MUSDB18HQDataset(root="/datasets/musdb18hq", split="test", sr=args.sample_rate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for i, (mono_cond, target) in enumerate(tqdm(test_loader)):
        cond_spec = audio2spec(mono_cond, device)  # (b, c, t, f)
        generated_spec = infer_and_generate_audio(model, diffusion, cond_spec)  # (b, c, t, f)
        output_audio = spec2audio(generated_spec, device).squeeze(0)  # (c, t)
        save_audio(audio=output_audio, output_path=args.output_gen_dir, it=i, sr=args.sample_rate)
        save_audio(audio=target.squeeze(0), output_path=args.output_gt_dir, it=i, sr=args.sample_rate)
        save_audio(audio=mono_cond.squeeze(0), output_path=args.output_gt_mono_dir, it=i, sr=args.sample_rate)

if __name__ == "__main__":
    main()
### how to use
'''
python sample_demo_m2s.py --checkpoint ./checkpoints/model_epoch_80000.pt \
                      --output_gt_dir ./sample_demos/gt \
                      --output_gt_mono_dir ./sample_demos/gt_mono \
                      --output_gen_dir ./sample_demos/gn \
                      --sample_rate 24000
'''