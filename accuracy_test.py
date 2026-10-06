"""
accuracy_test.py
-----------------
Task 5 - Accuracy Testing

Runs Whisper on a batch of test recordings and compares each transcript to
a human-verified "ground truth" transcript, reporting:

    - Word Error Rate (WER)
    - Accuracy % (1 - WER)
    - Substitutions / Insertions / Deletions (missing/incorrect words)

Usage:
    1. Put test recordings in:      sample_data/audio/<name>.wav
    2. Put matching ground truth in: sample_data/ground_truth/<name>.txt
       (same base filename, plain text of what was actually said)
    3. Run:
        python accuracy_test.py --model base

Target: >= 90% average accuracy across the test set (per Task 5 spec).
"""

import os
import re
import argparse
import glob

from utils import process_audio, run_whisper

try:
    import jiwer
except ImportError:
    jiwer = None


AUDIO_DIR = os.path.join(os.path.dirname(__file__), "sample_data", "audio")
GROUND_TRUTH_DIR = os.path.join(os.path.dirname(__file__), "sample_data", "ground_truth")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - so 'Hello!' and
    'hello' aren't counted as an error just from formatting differences."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> dict:
    """
    Compute Word Error Rate and breakdown of error types.

    WER = (Substitutions + Insertions + Deletions) / Number of words in reference
    Accuracy = (1 - WER) * 100
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if jiwer is not None:
        measures = jiwer.compute_measures(ref, hyp)
        wer = measures["wer"]
        return {
            "wer": wer,
            "accuracy_pct": max(0.0, (1 - wer)) * 100,
            "substitutions": measures["substitutions"],
            "insertions": measures["insertions"],
            "deletions": measures["deletions"],
            "hits": measures["hits"],
        }

    # --- Fallback: plain Levenshtein-based WER if jiwer isn't installed ---
    ref_words = ref.split()
    hyp_words = hyp.split()
    d = _levenshtein_ops(ref_words, hyp_words)
    total_errors = d["substitutions"] + d["insertions"] + d["deletions"]
    wer = total_errors / max(1, len(ref_words))
    return {
        "wer": wer,
        "accuracy_pct": max(0.0, (1 - wer)) * 100,
        "substitutions": d["substitutions"],
        "insertions": d["insertions"],
        "deletions": d["deletions"],
        "hits": len(ref_words) - d["substitutions"] - d["deletions"],
    }


def _levenshtein_ops(ref: list, hyp: list) -> dict:
    """Classic DP edit-distance, tracking op counts (used only if jiwer missing)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to classify ops
    i, j = n, m
    subs = ins = dels = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1
    return {"substitutions": subs, "insertions": ins, "deletions": dels}


def run_accuracy_suite(model_size: str = "base") -> None:
    audio_files = sorted(
        glob.glob(os.path.join(AUDIO_DIR, "*"))
    )

    if not audio_files:
        print(f"No test audio found in {AUDIO_DIR}")
        print("Add files like sample_data/audio/meeting1.wav plus a matching")
        print("sample_data/ground_truth/meeting1.txt and re-run.")
        return

    results = []

    print(f"Running accuracy test with Whisper model: '{model_size}'\n")

    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        gt_path = os.path.join(GROUND_TRUTH_DIR, base_name + ".txt")

        if not os.path.exists(gt_path):
            print(f"⚠️  Skipping '{base_name}' - no ground truth file found at {gt_path}")
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = f.read()

        print(f"▶ Processing: {base_name}")
        processed = process_audio(audio_path, work_dir="/tmp/accuracy_test_work")
        whisper_result = run_whisper(processed, model_size=model_size)
        hypothesis = whisper_result["text"]

        metrics = compute_wer(ground_truth, hypothesis)
        results.append({"name": base_name, **metrics})

        print(f"   Accuracy: {metrics['accuracy_pct']:.1f}%  "
              f"(WER: {metrics['wer']:.3f} | "
              f"Substitutions: {metrics['substitutions']}, "
              f"Insertions: {metrics['insertions']}, "
              f"Deletions: {metrics['deletions']})\n")

    if not results:
        print("No recordings had matching ground truth. Nothing to report.")
        return

    avg_accuracy = sum(r["accuracy_pct"] for r in results) / len(results)

    print("=" * 60)
    print("ACCURACY TEST SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:<30} {r['accuracy_pct']:>6.1f}%")
    print("-" * 60)
    print(f"  {'AVERAGE':<30} {avg_accuracy:>6.1f}%")
    print("=" * 60)

    target = 90.0
    if avg_accuracy >= target:
        print(f"✅ PASS - average accuracy {avg_accuracy:.1f}% meets the {target:.0f}% target.")
    else:
        print(f"❌ FAIL - average accuracy {avg_accuracy:.1f}% is below the {target:.0f}% target.")
        print("   Suggestions: try a larger Whisper model, check for noisy audio,")
        print("   or verify ground-truth transcripts are accurate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 5 - Whisper Accuracy Testing")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()
    run_accuracy_suite(model_size=args.model)
