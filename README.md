# DiffStereo: End-to-end Mono-to-Stereo Audio Generation with Diffusion Transformer
**DiffStereo** is an end-to-end diffusion transformer-based model for mono-to-stereo audio generation. Unlike traditional methods that require expert knowledge or explicit positional information to create stereo effects—limiting their scalability and generalization—DiffStereo explores this task without relying on any extra condition inputs. It directly synthesizes stereo audio from a mono waveform in an end-to-end fashion, requiring no human intervention or prior knowledge.

## DEMO 
🎧 Before you listen, put on your favorite headphones and make sure they sit nicely on both ears.<br>
👉 When you're ready, navigate to ./demos and enjoy the experience!


## Training 

## Step 1: Build Environment

Create conda environment and install the dependencies:

```bash
conda create -n m2s python=3.9
conda activate m2s
pip install -r requirement.txt
```

### Step 2: Adjust training-realted parameters in the configuration file
```
./config/train_spec.yaml
```

### Step 3: Start training
Use single GPU:
```bash
python train_spec.py
```
Use multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 --master_port=29602 train_spec_ddp.py
```

## Inference

Sample stereo audio using the trained model:

```bash
python sample_demo_m2s.py --checkpoint ./checkpoints/model_epoch_80000.pt \
                      --output_gt_dir ./sample_results/gt \
                      --output_gt_mono_dir ./sample_results/gt_mono \
                      --output_gen_dir ./sample_results/gn \
                      --sample_rate 24000
```


## To-Do
Further experienment:
- **VAE Integration**: Integrate a VAE to utilize latent representations for improving generation diversity and audio quality.


## External links
[1] DiT: https://github.com/facebookresearch/DiT


