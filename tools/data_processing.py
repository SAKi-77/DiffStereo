import os
import pdb
import torch
from einops import rearrange


def data_processing(
    data: dict,
    preprocess_func: callable, 
    device: str
):
    dataset_name = data["dataset_name"][0]

    if dataset_name == "MUSDB18HQ":

        # Condition: mono mixture
        mono_conditions = data["mixture"].mean(dim=1).unsqueeze(1).to(device)
        # Input: stereo mixture
        mixture = data["mixture"].to(device)
    else:
        raise NotImplementedError(dataset_name)

    # 'Preprocess function' is used when converting waveform to freq domain
    if preprocess_func is not None:
        # Target data
        x = preprocess_func(mixture, device)  # (b, c, t, f)
        # Condition
        cond_tf = preprocess_func(mono_conditions, device)  # (b, c, t, f)
    else:
        x = mixture  # (b, c, t)
        cond_tf = mono_conditions # (b, c, t)
 
    return x, cond_tf

