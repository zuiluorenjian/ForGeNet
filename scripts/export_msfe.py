"""Export an inference checkpoint with MSFE names and explicit model settings."""

import argparse
import hashlib
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from networks.checkpoints import canonical_msfe_state, infer_msfe_config, read_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.checkpoint), Path(args.output)
    if source.resolve() == target.resolve() or target.exists():
        parser.error("Choose a new output file; the source and existing files are never overwritten")
    checkpoint = read_checkpoint(source)
    state = canonical_msfe_state(checkpoint["model"])
    config = infer_msfe_config(checkpoint, state)
    exported = dict(format="forgenet-msfe-v1", model=state, model_config=config,
                    selection=checkpoint.get("selection"))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(exported, temporary)
        restored = read_checkpoint(temporary)["model"]
        if set(restored) != set(state) or any(not torch.equal(restored[key], value) for key, value in state.items()):
            raise RuntimeError("Exported model tensors differ from the source")
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with target.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    target.with_suffix(target.suffix + ".sha256").write_text(digest.hexdigest() + "  " + target.name + "\n")
    print(f"Exported {len(state)} unchanged tensors to {target}")
    print(f"SHA-256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
