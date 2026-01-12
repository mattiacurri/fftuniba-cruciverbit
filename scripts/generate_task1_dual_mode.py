#!/usr/bin/env python3
"""
Script to generate candidates for Task 1 in unconstrained and constrained modes.
Unconstrained: all generated candidates
Constrained: only candidates that respect the target length
"""

import argparse
import logging
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AddedToken, set_seed
import json
import numpy as np
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Genera candidati per Task 1 in modalità dual (unconstrained/constrained)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--input_file', type=str, default='hf_evalita2026/task_1/datasets/test_gold.csv',
                        help='Input CSV file with the clues')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for the results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
    parser.add_argument('--num_beams', type=int, default=100, help='Number of beams for beam search')
    parser.add_argument('--num_return_sequences', type=int, default=100,
                        help='Number of sequences to generate for each input')
    parser.add_argument('--max_retries', type=int, default=0,
                        help='Maximum number of attempts to generate sufficient candidates (constrained)')
    parser.add_argument('--min_candidates', type=int, default=50,
                        help='Minimum number of valid candidates required before retry')
    parser.add_argument('--special_tokens', type=str, default='true', help='Add special tokens for lengths (true/false)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for inference (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def extract_checkpoint_name(model_path: str) -> str:
    """Extracts the checkpoint name from the model path."""
    path_parts = Path(model_path).parts
    for part in reversed(path_parts):
        if 'checkpoint' in part:
            return part
    # If no checkpoint found, use the last folder
    return path_parts[-1] if path_parts else 'unknown'


def build_length_tokens(lengths):
    """Builds the special tokens for lengths."""
    tokens = []
    for length in sorted(set(lengths)):
        for tag in ("SL", "EL"):
            tokens.append(
                AddedToken(
                    f"[{tag} = {length}]",
                    lstrip=False,
                    rstrip=False,
                    normalized=False,
                )
            )
    return tokens


def ensure_special_tokens(tokenizer, model, special_tokens):
    """Adds special tokens to the tokenizer and resizes the model embeddings."""
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if added:
        logger.info(f"Added {added} new special tokens (vocab={len(tokenizer)})")
        model.resize_token_embeddings(len(tokenizer))
    else:
        logger.info("Special tokens already present in the tokenizer")
    return added


def generate_candidates(model, tokenizer, clues, answer_lengths, batch_size, num_beams, num_return_sequences, device, special_tokens):
    """
    Generates candidates for the clues using beam search.
    
    Returns:
        List of dicts with candidates and scores for each clue
    """
    model.eval()
    all_results = []
    
    # Add special tokens to the clues
    if special_tokens.lower() == 'true':
        clues_with_tokens = []
        for clue, length in zip(clues, answer_lengths):
            clue_with_token = f"indovinello: {clue} [SL = {length}]"
            clues_with_tokens.append(clue_with_token)
    else:
        clues_with_tokens = clues
    
    num_batches = (len(clues_with_tokens) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Pass 1"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(clues_with_tokens))
            batch_clues = clues_with_tokens[start_idx:end_idx]
            
            # Generate candidates for this batch
            batch_results = _generate_batch(
                model, tokenizer, batch_clues, num_beams, num_return_sequences, device
            )
            all_results.extend(batch_results)
    
    return all_results


def _generate_batch(model, tokenizer, batch_clues, num_beams, num_return_sequences, device):
    """Helper to generate a single batch."""
    inputs = tokenizer(batch_clues, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Remove token_type_ids if present (not supported by T5)
    inputs.pop('token_type_ids', None)
    
    outputs = model.generate(
        **inputs,
        max_length=32,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True
    )
    
    batch_results = []
    sequences_per_input = num_return_sequences
    
    for i in range(len(batch_clues)):
        start_seq = i * sequences_per_input
        end_seq = start_seq + sequences_per_input
        
        generated_sequences = outputs.sequences[start_seq:end_seq]
        decoded_candidates = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        # Strip whitespace from candidates
        decoded_candidates = [cand.strip() for cand in decoded_candidates]
        
        # Calculate the scores
        scores = []
        has_scores = hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None
        for seq_idx in range(len(generated_sequences)):
            global_seq_idx = start_seq + seq_idx
            if has_scores and global_seq_idx < len(outputs.sequences_scores):
                score = outputs.sequences_scores[global_seq_idx].item()
                scores.append(score)
            else:
                # Default score if not available
                scores.append(0.0)
        
        batch_results.append({
            'candidates': decoded_candidates,
            'scores': scores
        })
    
    return batch_results


def filter_by_length(candidates, scores, target_length):
    """
    Filters candidates keeping only those with the correct length.
    
    Returns:
        Tuple of (filtered_candidates, filtered_scores)
    """
    filtered_candidates = []
    filtered_scores = []
    
    for cand, score in zip(candidates, scores):
        # Strip and check length
        cand_stripped = cand.strip()
        if len(cand_stripped) == target_length:
            filtered_candidates.append(cand_stripped)
            filtered_scores.append(score)
    
    return filtered_candidates, filtered_scores


def calculate_metrics(df, k_values=[1, 5, 10, 100]):
    """
    Calculates accuracy@k and MRR.
    
    Args:
        df: DataFrame with 'rank' column
        k_values: List of k values for which to calculate accuracy
        
    Returns:
        Dict with metrics
    """
    metrics = {}
    
    # Accuracy@k
    for k in k_values:
        # An answer is correct if rank <= k
        correct = (df['rank'] > 0) & (df['rank'] <= k)
        acc = correct.sum() / len(df) * 100
        metrics[f'acc@{k}'] = round(acc, 2)
    
    # MRR (Mean Reciprocal Rank)
    # Consider only found answers (rank > 0)
    found = df[df['rank'] > 0]
    if len(found) > 0:
        mrr = (1.0 / found['rank']).mean()
        metrics['mrr'] = round(mrr, 4)
    else:
        metrics['mrr'] = 0.0
    
    return metrics


def save_results(df, output_path, filter_length=False):
    """
    Saves the results in CSV format.
    
    Args:
        df: DataFrame with the results
        output_path: Path of the output file
        filter_length: If True, filter only candidates with correct length
    """
    results = []
    has_answer = 'answer' in df.columns

    def sanitize_candidate(s: str) -> str:
        if not isinstance(s, str):
            return ''
        s = s.strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1]
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"[^A-Za-z]", "", s)
        return s.lower()
    
    for _, row in df.iterrows():
        candidates = row['candidates']
        scores = row['scores']
        # sanitize generated candidates
        sanitized_candidates = []
        sanitized_scores = []
        for idx, cand in enumerate(candidates):
            sc = sanitize_candidate(cand)
            if sc:
                sanitized_candidates.append(sc)
                sanitized_scores.append(scores[idx] if idx < len(scores) else 0.0)
        candidates = sanitized_candidates
        scores = sanitized_scores
        target_length = row['answer_length']
        
        # Filter by length if requested
        if filter_length:
            candidates, scores = filter_by_length(candidates, scores, target_length)
        
        result_row = {
            'clue': row['clue'],
            'answer_length': target_length,
            'candidates': ';'.join(candidates),
            'confidence_scores': ';'.join([f"{s:.6f}" for s in scores])
        }
        
        # Add 'answer' and 'rank' only if available (training mode)
        if has_answer:
            target_answer = row['answer'].lower()
            result_row['answer'] = row['answer']
            
            # Find the rank of the correct answer
            rank = -1
            for idx, cand in enumerate(candidates, 1):
                if cand.strip().lower() == target_answer:
                    rank = idx
                    break
            result_row['rank'] = rank
        
        results.append(result_row)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    return result_df


def print_metrics_report(metrics, mode):
    """Prints a formatted metrics report."""
    logger.info(f"\n{'='*60}")
    logger.info(f"METRICS - {mode.upper()}")
    logger.info(f"{'='*60}")
    
    # Accuracy@k
    acc_keys = sorted([k for k in metrics.keys() if k.startswith('acc@')],
                     key=lambda x: int(x.split('@')[1]))
    for key in acc_keys:
        k = key.split('@')[1]
        logger.info(f"  Accuracy@{k:>4s}: {metrics[key]:6.2f}%")
    
    # MRR
    if 'mrr' in metrics:
        logger.info(f"  MRR         : {metrics['mrr']:6.4f}")
    
    logger.info(f"{'='*60}\n")


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Extract the checkpoint name
    checkpoint_name = extract_checkpoint_name(args.model_path)
    logger.info(f"Checkpoint: {checkpoint_name}")
    
    # Convert path to absolute if relative
    model_path = Path(args.model_path).resolve()
    logger.info(f"Model path (resolved): {model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}...")
    from transformers import PreTrainedTokenizerFast
    
    # Load the tokenizer from the tokenizer.json file
    tokenizer_file = Path(model_path) / "tokenizer.json"
    if tokenizer_file.exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    else:
        raise FileNotFoundError(f"tokenizer.json not found in {model_path}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(device)
    
    # Load the data
    logger.info(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    logger.info(f"Found {len(df)} clues")
    
    clues = df['clue'].tolist()
    has_answers = 'answer' in df.columns
    logger.info(f"Mode: {'TRAINING (with answers)' if has_answers else 'TESTING (without answers)'}")
    answer_lengths = df['answer_length'].tolist()
    
    # Add special tokens
    if args.special_tokens.lower() == 'true':
        logger.info("Adding special tokens to tokenizer...")
        special_tokens = build_length_tokens(answer_lengths)
        ensure_special_tokens(tokenizer, model, special_tokens)
    else:
        logger.info("Special tokens not added (option disabled)")
    
    # Generate candidates
    logger.info(f"Generating candidates for {len(clues)} clues...")
    logger.info(f"Parameters: batch_size={args.batch_size}, num_beams={args.num_beams}, num_return_sequences={args.num_return_sequences}")
    
    results = generate_candidates(
        model, tokenizer, clues, answer_lengths,
        args.batch_size, args.num_beams, args.num_return_sequences, device, args.special_tokens.lower()
    )
    
    # Add the results to the dataframe
    df['candidates'] = [r['candidates'] for r in results]
    df['scores'] = [r['scores'] for r in results]
    
    # Output directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)
    
    # 1. UNCONSTRAINED: all candidates
    logger.info("\n" + "="*80)
    logger.info("SAVING UNCONSTRAINED RESULTS (all candidates)")
    logger.info("="*80)
    
    unconstrained_path = results_dir / f"task1_unconstrained_{checkpoint_name}.csv"
    df_unconstrained = save_results(df, unconstrained_path, filter_length=False)
    
    # Calculate unconstrained metrics (only if we have answers)
    if has_answers:
        metrics_unconstrained = calculate_metrics(df_unconstrained, k_values=[1, 5, 10, 100])
        print_metrics_report(metrics_unconstrained, "unconstrained")
    else:
        metrics_unconstrained = None
        logger.info("TESTING MODE: no metrics available (missing answers)")
    
    # 2. CONSTRAINED: only candidates with correct length
    logger.info("="*80)
    logger.info("SAVING CONSTRAINED RESULTS (filtered by length)")
    logger.info("="*80)
    
    constrained_path = results_dir / f"task1_constrained_{checkpoint_name}.csv"
    df_constrained = save_results(df, constrained_path, filter_length=True)
    
    # Calculate constrained metrics (only if we have answers)
    if has_answers:
        metrics_constrained = calculate_metrics(df_constrained, k_values=[1, 5, 10, 100])
        print_metrics_report(metrics_constrained, "constrained")
    else:
        metrics_constrained = None
        logger.info("TESTING MODE: no metrics available (missing answers)")
    
    # Save metrics to JSON
    metrics_json = {
        'unconstrained': metrics_unconstrained,
        'constrained': metrics_constrained,
        'config': {
            'model_path': args.model_path,
            'checkpoint': checkpoint_name,
            'batch_size': args.batch_size,
            'num_beams': args.num_beams,
            'num_return_sequences': args.num_return_sequences,
            'num_clues': len(df),
            'mode': 'training' if has_answers else 'testing'
        }
    }
    
    metrics_path = results_dir / f"task1_metrics_{checkpoint_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETED!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
