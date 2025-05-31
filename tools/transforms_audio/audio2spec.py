import torch
from torch import nn
from einops import rearrange


class AudioToSpec(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, audio, device):
        # audio: (b, c, t)

        # Extract spec feature.
        x = self.audio_to_spec(audio, device)  # (b, c*2, t, f)

        return x
    
    def audio_to_spec(self, audio, device):
        # audio: (b, c, t)

        channels = audio.shape[1]
        xs = []

        # Rescale audio to [-1, 1] if they are beyond [-1, 1].
        max_values = torch.max(torch.abs(audio), dim=2)[0]
        max_values[max_values < 1.] = 1
        audio /= max_values[:, :, None]
        
        for c in range(channels):
            x = torch.stft(
                input=audio[:, c, :],
                n_fft=1024, 
                hop_length=300, 
                win_length=1000,
                onesided=True,
                window=torch.hamming_window(1000).to(device),
                return_complex=True,
            )  # (b, f, t)
            # Calculate log spectrum
            x = self.log_spec(x)
            xs.append(x)
            
        # c2: Real & Imag for each channel of audio    
        xs = [rearrange(x, 'b f t c2 -> b c2 f t') for x in xs]
        x = torch.cat(xs, dim=1)  
        x = rearrange(x, 'b c f t -> b c t f')  
        return x

    
    def log_spec(self, spec):
        spec_factor = 0.15
        spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
        spec = spec * spec_factor
        
        spec = torch.view_as_real(spec)

        return spec
    
 
def check_for_nan(tensor, message=""):
    if torch.any(torch.isnan(tensor)):
        print(f"NaN detected: {message}")
    if torch.any(torch.isinf(tensor)):
        print(f"Inf detected: {message}")
