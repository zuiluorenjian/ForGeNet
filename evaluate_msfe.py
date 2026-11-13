"""Run batch evaluation on a labeled test folder."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader

from data.evaluation import ManifestDataset, folder_manifest
from networks.checkpoints import load_msfe_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="Test folder containing 0_real/1_fake, directly or inside generator subfolders")
    parser.add_argument("--transform", choices=["center_crop224", "resize224"], default="center_crop224")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = folder_manifest(args.data_root, args.transform)
    roots = {"data": args.data_root}
    model, audit = load_msfe_checkpoint(args.checkpoint, args.device)
    print(f"Loaded MSFE checkpoint ({audit['tensors_loaded']} tensors)", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(loading=audit, data_root=args.data_root,
                  threshold=0.5, precision="float32", datasets=[], status="running")
    for entry in manifest["datasets"]:
        dataset = ManifestDataset(entry, roots)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
        labels, scores = [], []
        started = time.monotonic()
        with torch.inference_mode():
            for images, target in loader:
                score = model(images.to(args.device, non_blocking=True)).sigmoid().flatten()
                scores.extend(score.cpu().tolist())
                labels.extend(target.tolist())
        labels, scores = np.asarray(labels), np.asarray(scores)
        predicted = scores > 0.5
        row = dict(name=entry["name"], transform=entry["transform"], count=len(labels),
                   acc=float(np.mean(predicted == labels)), ap=float(average_precision_score(labels, scores)),
                   r_acc=float(np.mean(predicted[labels == 0] == 0)),
                   f_acc=float(np.mean(predicted[labels == 1] == 1)),
                   elapsed_seconds=time.monotonic() - started)
        report["datasets"].append(row)
        if args.save_predictions:
            prediction_path = output.parent / (entry["name"].replace("/", "_") + "_predictions.npz")
            np.savez_compressed(prediction_path, labels=labels, scores=scores)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{row['name']}: ACC={100*row['acc']:.4f}% AP={100*row['ap']:.4f}%", flush=True)
    groups = {}
    for row in report["datasets"]:
        group = Path(args.data_root).name
        groups.setdefault(group, []).append(row)
    groups["overall"] = report["datasets"]
    report["summary"] = {name: dict(mAcc=100*float(np.mean([x["acc"] for x in rows])),
                                      mAP=100*float(np.mean([x["ap"] for x in rows])))
                         for name, rows in groups.items()}
    report["status"] = "complete"
    output.write_text(json.dumps(report, indent=2) + "\n")
    with output.with_suffix(".tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["datasets"][0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(report["datasets"])
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
