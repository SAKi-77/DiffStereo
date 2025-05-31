import torch
from torch import nn
from einops import rearrange

class SpecToAudio(nn.Module):
    def __init__(self):
        super().__init__()


    def __call__(self, audio, device):
        # audio: (b, c, t, f)

        # Extract spec feature.
        x = self.spec_to_audio(audio, device)  # (b, c, t)

        # Normalize x 
        x = self.normalize_audio(x)  # (b, c, t)

        return x
    
    def spec_to_audio(self, x, device):
        # x: (b, c*2, t, f)

        # Rearrange for iSTFT
        x = rearrange(x, 'b c t f -> b c f t')

        # Split two channels
        left_channel = x[:, :2, :, :]
        right_channel = x[:, 2:, :, :]

        left_channel  = rearrange(left_channel, 'b c f t -> b f t c')
        right_channel  = rearrange(right_channel, 'b c f t -> b f t c')

        # torch.istft requires complex input
        left_complex = torch.view_as_complex(left_channel.contiguous()) #[b, f, t]
        right_complex = torch.view_as_complex(right_channel.contiguous()) #[b, f, t]

        left_complex = self.log2spec(left_complex)
        right_complex = self.log2spec(right_complex)

        stereo_complex = torch.stack([left_complex, right_complex], dim=1)
        
        # List to hold audio for each channel
        audio_channels = []
        for c in range(stereo_complex.shape[1]):
            # Inverse STFT
            audio = torch.istft(
                input=stereo_complex[:, c],
                n_fft=1024,
                hop_length=300,
                win_length=1000,
                window=torch.hamming_window(1000).to(device),
                onesided=True,
            )
            audio_channels.append(audio)
        
        # Stack channels back to get (b, c, t) shape
        audio = torch.stack(audio_channels, dim=1)
        return audio
 
    def log2spec(self, spec):
        spec_factor = 0.15
        spec = (torch.exp(spec.abs())-1) * torch.exp(1j * spec.angle())
        spec = spec / spec_factor
        
        return spec

    def normalize_audio(self, x):
        return x / torch.max(torch.abs(x))
    
 
