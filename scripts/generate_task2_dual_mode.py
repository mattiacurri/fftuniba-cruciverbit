"""
Modular script to generate 2 CSVs for Task 2:
1. UNCONSTRAINED: all generated candidates (no filtering)
2. CONSTRAINED: candidates filtered by exact length (length-constrained)

Saves files in results/ with the checkpoint name in the filename.
Calculates acc@1,10,100 and MRR metrics for both.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple
import random

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Pattern to remove special tokens
SPECIAL_TOKEN_PATTERN = re.compile(r"\[SL\s*=\s*\d+\]|\[EL\s*=\s*\d+\]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dual-mode Task 2 predictions"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., it5_combined_dataset/checkpoint-430070)",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="hf_evalita2026/task_2/crosswords_datasets/test_gold_cross_clues.json",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save output CSV files",
    )
    parser.add_argument("--batch_size", type=int, default=42)
    parser.add_argument("--num_beams", type=int, default=100)
    parser.add_argument("--num_return_sequences", type=int, default=100)
    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help="Max retry attempts when constrained candidates are insufficient (0=disabled)",
    )
    parser.add_argument(
        "--min_candidates",
        type=int,
        default=50,
        help="Minimum candidates required for constrained mode before retry",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def clean_answer(text: str) -> str:
    """Cleans the text by removing special tokens and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.replace("<pad>", "").replace("</s>", "").replace("<s>", "")
    text = re.sub(SPECIAL_TOKEN_PATTERN, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def calculate_metrics(
    targets: List[str],
    candidates_list: List[List[str]],
    k_values: List[int] = [1, 5, 10, 100, 250, 500, 1000],
) -> Dict[str, float]:
    """Calculates accuracy and MRR metrics."""
    stats = {f"acc@{k}": 0 for k in k_values}
    stats["mrr"] = 0.0
    total = len(targets)

    for target, candidates in zip(targets, candidates_list):
        target = str(target).lower().strip()

        # Find rank of the correct answer
        hit_rank = None
        for i, cand in enumerate(candidates):
            if cand == target:
                hit_rank = i + 1
                break

        if hit_rank:
            stats["mrr"] += 1.0 / hit_rank
            for k in k_values:
                if hit_rank <= k:
                    stats[f"acc@{k}"] += 1

    # Normalize
    for k in stats:
        stats[k] /= total

    return stats


def extract_checkpoint_name(model_path: str) -> str:
    """Extracts the checkpoint name from the path (e.g., checkpoint-430070)."""
    path_obj = Path(model_path)
    if path_obj.name.startswith("checkpoint-"):
        return path_obj.name
    # If the path contains checkpoint in the parent directory
    for part in path_obj.parts:
        if part.startswith("checkpoint-"):
            return part
    # Fallback: use the last component of the path
    return path_obj.name


def generate_candidates(
    model, tokenizer, data: List[Dict], args, device: str
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Generates candidates using beam search.
    With retry for clues that have few candidates of the correct length.

    Returns:
        all_candidates: List of lists of candidates (cleaned)
        all_confidences: List of lists of confidence scores
    """
    all_candidates = []
    all_confidences = []

    batch_size = args.batch_size
    total_samples = len(data)

    LOGGER.info(f"Generating candidates for {total_samples} clues...")

    # First pass: generate for all
    for i in tqdm(range(0, total_samples, batch_size), desc="Pass 1"):
        batch_data = data[i : i + batch_size]
        batch_candidates, batch_scores = _generate_batch(
            model, tokenizer, batch_data, args, device
        )
        all_candidates.extend(batch_candidates)
        all_confidences.extend(batch_scores)

    # Retry for clues with few constrained candidates
    retry_indices = []
    if args.max_retries > 0:
        for idx, item in enumerate(data):
            target_len = item["length"]
            constrained_count = sum(
                1 for c in all_candidates[idx] if len(c) == target_len
            )
            if constrained_count < args.min_candidates:
                retry_indices.append(idx)

    if retry_indices:
        LOGGER.info(
            f"Retry for {len(retry_indices)} clues with few constrained candidates..."
        )

        for retry_attempt in range(args.max_retries):
            still_need_retry = []

            # Process retries in batches too
            for batch_start in tqdm(
                range(0, len(retry_indices), batch_size),
                desc=f"Retry {retry_attempt + 1}/{args.max_retries}",
            ):
                batch_retry_idx = retry_indices[batch_start : batch_start + batch_size]
                batch_data = [data[idx] for idx in batch_retry_idx]

                batch_candidates, batch_scores = _generate_batch(
                    model, tokenizer, batch_data, args, device
                )

                # Merge new candidates with existing ones
                for local_idx, global_idx in enumerate(batch_retry_idx):
                    target_len = data[global_idx]["length"]

                    # Merge candidates (remove duplicates)
                    existing_cands = all_candidates[global_idx]
                    existing_scores = all_confidences[global_idx]
                    new_cands = batch_candidates[local_idx]
                    new_scores = batch_scores[local_idx]

                    cand_dict = {c: s for c, s in zip(existing_cands, existing_scores)}
                    for c, s in zip(new_cands, new_scores):
                        if c not in cand_dict:
                            cand_dict[c] = s
                        else:
                            cand_dict[c] = max(cand_dict[c], s)  # Keep the best score

                    # Reorder by score
                    sorted_items = sorted(
                        cand_dict.items(), key=lambda x: x[1], reverse=True
                    )
                    all_candidates[global_idx] = [c for c, s in sorted_items]
                    all_confidences[global_idx] = [s for c, s in sorted_items]

                    # Check if retry is still needed
                    constrained_count = sum(
                        1 for c in all_candidates[global_idx] if len(c) == target_len
                    )
                    if constrained_count < args.min_candidates:
                        still_need_retry.append(global_idx)

            retry_indices = still_need_retry
            if not retry_indices:
                LOGGER.info(
                    f"All candidates meet requirements after {retry_attempt + 1} retries"
                )
                break

        if retry_indices:
            LOGGER.warning(
                f"{len(retry_indices)} clues still below min_candidates after all retries"
            )

    return all_candidates, all_confidences


def _generate_batch(
    model, tokenizer, batch_data: List[Dict], args, device: str
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Generates candidates for a single batch.
    """
    # Prepare prompts
    prompts = [
        f"indovinello: {item['clue']} [SL = {item['length']}]" for item in batch_data
    ]

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    # Calculate log probabilities
    if hasattr(outputs, "sequences_scores"):
        scores = outputs.sequences_scores.cpu().numpy()
    else:
        scores = np.zeros(len(decoded_preds))

    num_return = args.num_return_sequences
    batch_candidates = []
    batch_scores = []

    for j in range(len(batch_data)):
        start_idx = j * num_return
        end_idx = (j + 1) * num_return

        batch_preds = decoded_preds[start_idx:end_idx]
        batch_conf = (
            scores[start_idx:end_idx] if len(scores) > 0 else [0.0] * num_return
        )

        # Clean predictions
        cleaned_preds = [clean_answer(p) for p in batch_preds]

        # Remove duplicates
        seen = {}
        unique_candidates = []
        unique_scores = []

        for pred, score in zip(cleaned_preds, batch_conf):
            if pred and pred not in seen:
                seen[pred] = score
                unique_candidates.append(pred)
                unique_scores.append(float(score))

        batch_candidates.append(unique_candidates)
        batch_scores.append(unique_scores)

    return batch_candidates, batch_scores


def filter_by_length(
    candidates: List[str], scores: List[float], target_length: int
) -> Tuple[List[str], List[float]]:
    """Filters candidates by exact length."""
    filtered_cands = []
    filtered_scores = []

    for cand, score in zip(candidates, scores):
        if len(cand) == target_length:
            filtered_cands.append(cand)
            filtered_scores.append(score)

    return filtered_cands, filtered_scores


def sanitize_candidate(s: str) -> str:
    """Sanitize candidate: remove surrounding quotes, bracketed tokens and keep only A-Za-z characters."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # Remove surrounding quotes
    if len(s) >= 2 and (
        (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
    ):
        s = s[1:-1]
    # Remove bracket tokens e.g. [SL = 5]
    s = re.sub(r"\[.*?\]", "", s)
    # Keep only ASCII letters
    s = re.sub(r"[^A-Za-z]", "", s)
    return s.lower()


def save_results(
    data: List[Dict],
    all_candidates: List[List[str]],
    all_confidences: List[List[float]],
    output_path: str,
    filter_length: bool = False,
):
    """
    Saves results to CSV.

    Args:
        filter_length: If True, filters candidates by length (CONSTRAINED mode)
    """
    results = []

    for i, item in enumerate(data):
        candidates = all_candidates[i]
        confidences = all_confidences[i]
        # Sanitize candidates right away
        sanitized_candidates = []
        sanitized_confidences = []
        for idx, cand in enumerate(candidates):
            sc = sanitize_candidate(cand)
            if sc:
                sanitized_candidates.append(sc)
                # keep corresponding confidence if exists
                sanitized_confidences.append(
                    confidences[idx] if idx < len(confidences) else 0.0
                )
        candidates = sanitized_candidates
        confidences = sanitized_confidences
        target_length = item["length"]

        # Apply length filter if requested
        if filter_length:
            candidates, confidences = filter_by_length(
                candidates, confidences, target_length
            )

        # Format candidates and confidence scores
        candidates_str = ";".join(candidates)
        confidence_str = ";".join([f"{s:.6f}" for s in confidences])

        # Find rank of the correct answer (if available)
        target = item.get("target", item.get("answer", ""))
        target_sanitized = sanitize_candidate(target) if target else ""
        rank = -1
        if target:
            for idx, cand in enumerate(candidates):
                if cand == target_sanitized:
                    rank = idx + 1
                    break

        results.append(
            {
                "crossword_num": item["num_cruciverba"],
                "clue_num": item["num"],
                "direction": "Horizontal"
                if item.get("direction", "A") == "A"
                else "Vertical",
                "clue": item["clue"],
                "real_answer": target if target else "",
                "answer_length": item["length"],
                "candidates": candidates_str,
                "confidence_scores": confidence_str,
                "rank": rank,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved: {output_path}")

    return df


def print_metrics_report(
    metrics_unconstrained: Dict, metrics_constrained: Dict, checkpoint_name: str
):
    """Prints formatted metrics report."""
    print("\n" + "=" * 70)
    print(f"METRICS REPORT - {checkpoint_name}")
    print("=" * 70)
    print(f"{'Metric':<20} | {'UNCONSTRAINED':<20} | {'CONSTRAINED':<20}")
    print("-" * 70)

    # Sort metrics
    keys = sorted(
        metrics_unconstrained.keys(), key=lambda k: (0 if "acc" in k else 1, k)
    )

    for k in keys:
        v_unc = metrics_unconstrained[k]
        v_con = metrics_constrained[k]
        print(f"{k:<20} | {v_unc:>20.4f} | {v_con:>20.4f}")

    print("=" * 70)
    print("\nLEGEND:")
    print("  UNCONSTRAINED: All generated candidates (no filtering)")
    print("  CONSTRAINED:   Candidates filtered by exact length")
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    device = args.device

    # Set seed for reproducibility
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Extract checkpoint name
    checkpoint_name = extract_checkpoint_name(args.model_path)
    LOGGER.info(f"Checkpoint: {checkpoint_name}")

    # Load model
    LOGGER.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()

    # Load data
    LOGGER.info(f"Loading data from {args.input_jsonl}...")
    data = []
    with open(args.input_jsonl, "r") as f:
        for i, line in enumerate(f):
            grid_clues = json.loads(line)
            for j, item in enumerate(grid_clues):
                item["num_cruciverba"] = i
                item["num"] = j
                data.append(item)

    LOGGER.info(f"Found {len(data)} clues")

    # Generate candidates (only once)
    all_candidates, all_confidences = generate_candidates(
        model, tokenizer, data, args, device
    )

    # Prepare output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    unconstrained_path = output_dir / f"task2_unconstrained_{checkpoint_name}.csv"
    constrained_path = output_dir / f"task2_constrained_{checkpoint_name}.csv"

    # Save UNCONSTRAINED (WITHOUT length filter - all candidates)
    LOGGER.info("Creating UNCONSTRAINED CSV (all candidates)...")
    df_unconstrained = save_results(
        data,
        all_candidates,
        all_confidences,
        str(unconstrained_path),
        filter_length=False,
    )

    # Save CONSTRAINED (WITH length filter)
    LOGGER.info("Creating CONSTRAINED CSV (length-filtered)...")
    df_constrained = save_results(
        data, all_candidates, all_confidences, str(constrained_path), filter_length=True
    )

    # Calculate metrics
    LOGGER.info("Calculating metrics...")

    # Prepare lists for metrics calculation (only if we have targets)
    has_targets = all(item.get("target") or item.get("answer") for item in data)

    if has_targets:
        targets = [
            str(item.get("target", item.get("answer", ""))).lower().strip()
            for item in data
        ]

        # UNCONSTRAINED metrics
        unconstrained_cands = []
        for _, row in df_unconstrained.iterrows():
            cands = str(row["candidates"]).split(";") if row["candidates"] else []
            unconstrained_cands.append(cands)

        metrics_unconstrained = calculate_metrics(targets, unconstrained_cands)

        # CONSTRAINED metrics
        constrained_cands = []
        for _, row in df_constrained.iterrows():
            cands = str(row["candidates"]).split(";") if row["candidates"] else []
            constrained_cands.append(cands)

        metrics_constrained = calculate_metrics(targets, constrained_cands)

        # Print report
        print_metrics_report(
            metrics_unconstrained, metrics_constrained, checkpoint_name
        )

        # Save metrics to JSON
        metrics_path = output_dir / f"task2_metrics_{checkpoint_name}.json"
        metrics_data = {
            "checkpoint": checkpoint_name,
            "model_path": args.model_path,
            "num_clues": len(data),
            "unconstrained": metrics_unconstrained,
            "constrained": metrics_constrained,
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        LOGGER.info(f"Metrics saved to: {metrics_path}")
    else:
        LOGGER.info("No targets available - skip metrics calculation")

    LOGGER.info("✅ Completed!")


if __name__ == "__main__":
    main()
