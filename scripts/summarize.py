"""Compute means and sample standard deviations from completed evaluations."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in args.results]
    if any(report.get("status") != "complete" for report in reports):
        parser.error("All evaluations must be complete")
    identifiers = [Path(report["loading"]["checkpoint"]).stem for report in reports]
    groups = set.intersection(*(set(report["summary"]) for report in reports))
    summary = {}
    for group in sorted(groups):
        summary[group] = {}
        for metric in ("mAcc", "mAP"):
            values = [report["summary"][group][metric] for report in reports]
            summary[group][metric] = dict(mean=statistics.mean(values),
                                          sample_std=statistics.stdev(values) if len(values) > 1 else None)
    print(json.dumps(dict(checkpoints=identifiers, count=len(reports), summary=summary), indent=2))


if __name__ == "__main__":
    main()
