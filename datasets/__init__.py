def get_dataset(cfg: dict):

    if cfg["name"] == "GTZAN":

        from audidata.datasets import GTZAN
        from audidata.io.crops import RandomCrop

        train_dataset = GTZAN(
            root=cfg["root"],
            split="train",
            test_fold=0,
            sr=cfg["sample_rate"],
            crop=RandomCrop(clip_duration=cfg["clip_duration"])
        )

    elif cfg["name"] == "Shutterstock":

        from audidata.datasets import Shutterstock
        from audidata.io.crops import RandomCrop

        train_dataset = Shutterstock(
            root=cfg["root"],
            sr=cfg["sample_rate"],
            crop=RandomCrop(clip_duration=cfg["clip_duration"])
        )

    elif cfg["name"] == "MUSDB18HQ":

        from audidata.datasets import MUSDB18HQ
        from audidata.io.crops import RandomCrop
    
        train_dataset = MUSDB18HQ(
            root=cfg["root"],
            split="train",
            sr=cfg["sample_rate"],
            crop=RandomCrop(clip_duration=cfg["clip_duration"])    
        ) #remix={"no_remix": 1.0, "half_remix": 0.0, "full_remix": 0.}

    elif cfg["name"] == "MAESTRO":

        from audidata.datasets import MAESTRO
        from audidata.io.crops import RandomCrop
        from audidata.transforms.midi import PianoRoll

        train_dataset = MAESTRO(
            root=cfg["root"],
            split="train",
            sr=cfg["sample_rate"],
            crop=RandomCrop(clip_duration=cfg["clip_duration"], end_pad=0.),
            target_transform=PianoRoll(fps=cfg["fps"], pitches_num=128),
        )

    else:
        raise NotImplementedError(cfg["name"])

    return train_dataset


