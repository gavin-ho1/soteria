"""Audit warm-up resampling and temporal-density cues in rolling replay"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import data as cdata
from common.paths import resolve_project_path
from streaming_predictor.config import Config
from streaming_predictor.data import expand_view_instances, select_eval_view_instances
from streaming_predictor.rolling_evaluation import build_rolling_windows, replay_end_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, _ = resolve_project_path(args.output, field="--output")
    window_sec = Config().model.window_sec
    num_tokens = Config().model.num_tokens

    sequences = cdata.load_view_sequences(verbose=False)
    seizures = cdata.load_seizures(sequences, verbose=False)
    by_ssid = {seizure.ssid: seizure for seizure in seizures}
    instances = select_eval_view_instances(
        expand_view_instances([seizure.ssid for seizure in seizures], by_ssid)
    )

    records = []
    for instance in instances:
        _, times = cdata._frames(sequences[f"{instance.ssid}_{instance.view}"])
        end_time = replay_end_time(times, label=instance.label, onset=instance.onset_sec)
        windows = build_rolling_windows(
            times,
            end_time=end_time,
            window_sec=window_sec,
            cadence_sec=1.0,
            num_tokens=num_tokens,
        )
        repeated = [len(np.unique(window.sample_indices)) < num_tokens for window in windows]
        short_history = [
            window.decision_time - float(times[0]) < window_sec - 1e-9
            for window in windows
        ]
        records.append({
            "ssid": instance.ssid,
            "label": int(instance.label),
            "n_decisions": len(windows),
            "n_repeated_token_decisions": int(sum(repeated)),
            "n_under_20s_history_decisions": int(sum(short_history)),
            "never_reaches_20s_history": bool(all(short_history)),
        })

    def summarize(label: int | None) -> dict:
        selected = records if label is None else [r for r in records if r["label"] == label]
        decisions = sum(r["n_decisions"] for r in selected)
        repeated = sum(r["n_repeated_token_decisions"] for r in selected)
        under_20 = sum(r["n_under_20s_history_decisions"] for r in selected)
        never_20 = sum(r["never_reaches_20s_history"] for r in selected)
        return {
            "recordings": len(selected),
            "decisions": decisions,
            "repeated_token_decisions": repeated,
            "repeated_token_decision_fraction": repeated / decisions,
            "under_20s_history_decisions": under_20,
            "under_20s_history_decision_fraction": under_20 / decisions,
            "recordings_never_reaching_20s": never_20,
            "recordings_never_reaching_20s_fraction": never_20 / len(selected),
        }

    payload = {
        "scope": "deterministic evaluation view; one-second causal rolling replay",
        "window_sec": window_sec,
        "num_tokens": num_tokens,
        "summary": {
            "all": summarize(None),
            "onset_positive": summarize(1),
            "onset_negative": summarize(0),
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2))
    pending.replace(output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
