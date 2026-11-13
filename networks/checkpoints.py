"""Strict loading of released MSFE detectors, including legacy parameter names."""

from collections import OrderedDict
from pathlib import Path

import torch


def read_checkpoint(path):
    path = Path(path)
    # Historical files contain argparse metadata as well as tensors. Only load
    # trusted model files. mmap avoids copying optimizer state during inference.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def canonical_msfe_state(state):
    """Rename legacy Mona keys without changing or dropping any tensor."""
    result = OrderedDict()
    for key, tensor in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        key = key.replace(".mona_adapter.", ".msfe_adapter.")
        key = key.replace(".mona_op.", ".msfe_op.")
        result[key] = tensor
    return result


def infer_msfe_config(checkpoint, state):
    adapters = sorted(
        (int(key.split(".")[2]), key)
        for key in state
        if key.startswith("visual.resblocks.")
        and key.endswith(".msfe_adapter.project1.weight")
    )
    if not adapters:
        raise ValueError("Expected a Stage-2 MSFE checkpoint, not a Stage-1 or linear-only file")
    indices = [index for index, _ in adapters]
    prefix = adapters[0][1].removesuffix("project1.weight")
    convs = sorted(
        (int(key.split(".")[-2]), value.shape[-1])
        for key, value in state.items()
        if key.startswith(prefix + "msfe_op.depthwise.") and key.endswith(".weight")
    )
    options = checkpoint.get("opt", {})
    if not isinstance(options, dict):
        options = vars(options)
    width = state[adapters[0][1]].shape[1]
    patch = state["visual.conv1.weight"].shape[-1]
    inferred_names = {(1024, 14): "ViT-L/14", (768, 16): "ViT-B/16", (768, 32): "ViT-B/32"}
    config = dict(
        clip_model_name=options.get("arch", inferred_names.get((width, patch), "")).removeprefix("CLIP:"),
        num_classes=state["classifier.1.weight"].shape[0], use_msfe=True,
        msfe_hidden_dim=state[adapters[0][1]].shape[0],
        msfe_kernel_sizes=[size for _, size in convs],
        msfe_dropout=options.get("msfe_dropout", options.get("mona_dropout", 0.1)),
        msfe_scale=options.get("msfe_scale", options.get("mona_scale", 0.1)),
        num_adapted_layers=len(indices), readout="cls",
    )
    config.update(checkpoint.get("model_config", {}))
    if config["readout"] != "cls" or state["classifier.1.weight"].shape[1] != state["visual.proj"].shape[1]:
        raise ValueError("This release reproduces CLS-only checkpoints; patch/concat experiments need their own model")
    return config


def load_msfe_checkpoint(path, device="cpu"):
    from models.msfe_clip import MSFECLIPModel

    checkpoint = read_checkpoint(path)
    state = canonical_msfe_state(checkpoint["model"])
    config = infer_msfe_config(checkpoint, state)
    clip_state = {key[len("original_clip."):]: tensor for key, tensor in state.items()
                  if key.startswith("original_clip.")}
    constructor = {key: value for key, value in config.items() if key != "readout"}
    model = MSFECLIPModel(**constructor, clip_state_dict=clip_state)
    incompatible = model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    audit = dict(
        checkpoint=str(Path(path).resolve()), model_config=config,
        tensors_loaded=len(state), missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        selection=checkpoint.get("selection"),
        legacy_names_converted=any(".mona_adapter." in key for key in checkpoint["model"]),
    )
    return model, audit


def load_stage1_visual(model, path):
    """Transfer every merged Stage-1 visual tensor, including all Q/V updates."""
    checkpoint = read_checkpoint(path)
    state = checkpoint.get("model_state_dict", checkpoint)
    prefix = "encoder.model.visual."
    source = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    expected = {key for key in model.visual.state_dict() if ".msfe_adapter." not in key}
    mapped = {}
    for key, value in source.items():
        if key.startswith("transformer.resblocks."):
            index, suffix = key[len("transformer.resblocks."):].split(".", 1)
            key = f"resblocks.{index}.original_block.{suffix}"
        mapped[key] = value
    if set(mapped) != expected:
        raise ValueError(f"Incomplete Stage-1 visual transfer; missing={sorted(expected-set(mapped))}, "
                         f"unexpected={sorted(set(mapped)-expected)}")
    model.visual.load_state_dict(mapped, strict=False)  # Only MSFE parameters are intentionally absent.
    return dict(visual_tensors_loaded=len(mapped), merged_lora_loaded=True)
