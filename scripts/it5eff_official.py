"""Fine-tune gsarti/it5-efficient-small-el32 with bracketed length tokens on Evalita 2026 Task 1."""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AddedToken,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)
# Pattern to recognize special length tokens in the text
SPECIAL_TOKEN_PATTERN = re.compile(r'\[SL\s*=\s*\d+\]|\[EL\s*=\s*\d+\]')


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune IT5 Efficient with special length tokens")
    
    # --- MODIFICATION 1: Efficient model by default ---
    parser.add_argument("--model_name", type=str, default="gsarti/it5-efficient-small-el32")
    
    parser.add_argument("--dataset_dir", type=Path, default=Path("evalita2026/task_1/datasets"))
    parser.add_argument("--output_dir", type=Path, default=Path("tmp_it5_efficient_riddles_nodict_notokens"))
    parser.add_argument("--cache_dir", type=Path, default=Path("cached_tokenized"))
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    
    # --- MODIFICATION 2: Batch Size doubled (the model is lighter) ---
    parser.add_argument("--train_batch_size", type=int, default=32) 
    parser.add_argument("--eval_batch_size", type=int, default=32)
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_limit", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--subset", type=int, default=0, help="Optional number of training examples for quick tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocessing_workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--num_generations", type=int, default=10, help="Number of generations per example for evaluation")
    parser.add_argument("--generation_log_samples", type=int, default=200, help="Number of examples to log generations for")
    parser.add_argument("--skip_initial_eval", action="store_true", help="Skip evaluation before training")
    parser.add_argument("--use_dictionary", type=lambda x: x.lower() == 'true', default=False, help="Use Italian dictionary for data augmentation (true/false)")
    parser.add_argument("--no_special_tokens", action="store_true", help="Disable special length tokens [SL/EL] for ablation study")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_all_datasets(dataset_dir: Path, use_dictionary: bool = False) -> Tuple[Dataset, Dataset]:
    # Load local Evalita dataset
    train_csv = dataset_dir / "train.csv"
    val_csv = dataset_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        # Fallback if local files don't exist, useful for quick test
        LOGGER.warning(f"CSV files not found in {dataset_dir}. Creating dummy data if they don't exist.")
        # If you have the files, remove this block or ensure paths are correct
        if not train_csv.exists():
            raise FileNotFoundError(f"Missing train/val CSV files under {dataset_dir}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    for frame in (train_df, val_df):
        frame["clue"] = frame["clue"].fillna("")
        frame["answer"] = frame["answer"].fillna("")
        if "answer_length" not in frame.columns:
            frame["answer_length"] = frame["answer"].apply(lambda x: len(str(x)))
        frame["answer_length"] = frame["answer_length"].astype(int)

    evalita_train = Dataset.from_pandas(train_df)
    evalita_val = Dataset.from_pandas(val_df)

    cols_to_keep = ["clue", "answer", "answer_length"]
    evalita_train = evalita_train.select_columns(cols_to_keep)
    evalita_val = evalita_val.select_columns(cols_to_keep)

    # Load external dictionary for data augmentation
    if not use_dictionary:
        LOGGER.info("Skipping dictionary augmentation (--use_dictionary not set)")
        return evalita_train, evalita_val
    
    LOGGER.info("Loading mik3ml/italian-dictionary...")
    try:
        dict_dataset = load_dataset("mik3ml/italian-dictionary", split="train")
        LOGGER.info("Dictionary dataset columns: %s", dict_dataset.column_names)
        
        rename_map = {}
        if "definition" in dict_dataset.column_names:
            rename_map["definition"] = "clue"
        if "word" in dict_dataset.column_names:
            rename_map["word"] = "answer"
        
        if rename_map:
            dict_dataset = dict_dataset.rename_columns(rename_map)
        
        if "clue" in dict_dataset.column_names and "answer" in dict_dataset.column_names:
            dict_dataset = dict_dataset.select_columns(["clue", "answer"])
            
            def add_length(batch):
                return {"answer_length": [len(str(a)) if a else 0 for a in batch["answer"]]}
            
            dict_dataset = dict_dataset.map(
                add_length, 
                batched=True, 
                batch_size=1000,
                desc="Processing dictionary dataset"
            )
            
            LOGGER.info("Merging Evalita train (%d) with Dictionary train (%d)", len(evalita_train), len(dict_dataset))
            combined_train = concatenate_datasets([evalita_train, dict_dataset])
        else:
            combined_train = evalita_train
    except Exception as e:
        LOGGER.warning(f"Unable to load additional dictionary: {e}. Proceeding only with Evalita.")
        combined_train = evalita_train

    return combined_train, evalita_val


def collect_lengths(datasets: Iterable[Dataset]) -> List[int]:
    lengths = set()
    for split in datasets:
        if "answer_length" in split.column_names:
            lengths.update(int(l) for l in split["answer_length"] if l)
    if not lengths:
        lengths = {1}
    minimo, massimo = min(lengths), max(lengths)
    lengths = list(range(minimo, massimo + 1))
    return sorted(lengths)


def build_length_tokens(lengths: Iterable[int]) -> List[AddedToken]:
    tokens: List[AddedToken] = []
    for length in sorted(set(lengths)):
        for tag in ("SL", "EL"):
            # Use AddedToken with normalize=False to prevent the tokenizer from splitting it
            tokens.append(
                AddedToken(
                    f"[{tag} = {length}]",
                    lstrip=False,
                    rstrip=False,
                    normalized=False,
                )
            )
    return tokens


def ensure_special_tokens(tokenizer, model, special_tokens: List[AddedToken]) -> None:
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if added:
        LOGGER.info("Added %d new special tokens (vocab=%d)", added, len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))


def _format_target(answer: str, length: int, use_special_tokens: bool = True) -> str | None:
    answer = str(answer).strip()
    if not answer:
        return None
    if not use_special_tokens:
        return answer
    length = int(length) if length else len(answer)
    return f"[SL = {length}] {answer} [EL = {length}]"


def preprocess_batch(
    examples,
    tokenizer,
    max_input_length: int,
    max_target_length: int,
    use_special_tokens: bool = True,
):
    clues = examples["clue"]
    answers = examples["answer"]
    lengths = examples.get("answer_length") or [None] * len(answers)

    model_inputs = []
    targets = []

    for clue, answer, length in zip(clues, answers, lengths):
        clue = str(clue).strip()
        target = _format_target(answer, length, use_special_tokens)
        if not clue or target is None:
            continue
        
        real_length = int(length) if length else len(str(answer).strip())
        # Prompt optimized for Italian model
        if use_special_tokens:
            model_inputs.append(f"indovinello: {clue} [SL = {real_length}]")
        else:
            model_inputs.append(f"indovinello: {clue}")
        targets.append(target)

    if not model_inputs:
        return {}

    tokenized_inputs = tokenizer(
        model_inputs,
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )
    tokenized_targets = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )

    labels = tokenized_targets["input_ids"]
    pad_id = tokenizer.pad_token_id
    # Replace padding with -100 to ignore it in loss calculation
    labels = [[-100 if token == pad_id else token for token in label] for label in labels]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def _clean_answer(text: str) -> str:
    text = re.sub(SPECIAL_TOKEN_PATTERN, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def log_generation_metrics(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generation_metrics.jsonl"
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dump_generation_samples(output_dir: Path, tag: str, samples: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"generation_samples_{tag}.jsonl"
    with sample_path.open("w", encoding="utf-8") as fp:
        for sample in samples:
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")


def evaluate_with_generations(
    model,
    tokenizer,
    examples: list[dict],
    num_generations: int,
    max_input_length: int,
    max_target_length: int,
    batch_size: int,
    output_dir: Path,
    tag: str,
    num_beams: int,
    max_logged_examples: int = 100,
):
    if not examples or num_generations <= 0:
        return {}

    device = next(model.parameters()).device
    total = len(examples)
    batch_size = max(1, batch_size)
    beams = max(num_beams, num_generations)

    counts = {"acc1": 0, "acc5": 0, "acc10": 0}
    reciprocal_rank_sum = 0.0
    logged_samples: list[dict] = []
    
    # Ensure the model is in eval mode
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for start in range(0, total, batch_size):
            if (start % (batch_size * 50)) == 0:
                LOGGER.info(f"Generation progress: {start}/{total} examples")
            batch = examples[start:start + batch_size]
            prompts = []
            answers = []
            lengths = []
            for item in batch:
                clue = str(item.get("clue", ""))
                answer = str(item.get("answer", ""))
                length = int(item.get("answer_length") or len(answer))
                prompt = f"indovinello: {clue} [SL = {length}]"
                prompts.append(prompt)
                answers.append(answer)
                lengths.append(length)

            inputs = tokenizer(
                prompts,
                max_length=max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generation
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_target_length + 4,
                num_beams=beams,
                num_return_sequences=num_generations,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )

            outputs = outputs.cpu()
            for idx, (prompt, gold_answer, length) in enumerate(zip(prompts, answers, lengths)):
                gold_clean = _clean_answer(gold_answer)
                candidate_cleans = []
                raw_candidates = []
                
                # Decode the beams
                for rank in range(num_generations):
                    seq_index = idx * num_generations + rank
                    if seq_index >= outputs.size(0):
                        break
                    decoded = tokenizer.decode(outputs[seq_index], skip_special_tokens=False)
                    raw_candidates.append(decoded)
                    candidate_cleans.append(_clean_answer(decoded))

                if not candidate_cleans:
                    continue

                # Calculate metrics for single example
                if any(gold_clean in cand for cand in candidate_cleans[:1]):
                    counts["acc1"] += 1
                if any(gold_clean in cand for cand in candidate_cleans[:min(5, len(candidate_cleans))]):
                    counts["acc5"] += 1
                if any(gold_clean in cand for cand in candidate_cleans[:min(10, len(candidate_cleans))]):
                    counts["acc10"] += 1

                match_rank = None
                for pos, cand in enumerate(candidate_cleans, start=1):
                    if gold_clean in cand:
                        match_rank = pos
                        reciprocal_rank_sum += 1.0 / pos
                        break

                if len(logged_samples) < max_logged_examples:
                    logged_samples.append({
                        "clue": prompt,
                        "answer": gold_answer,
                        "length": length,
                        "match_rank": match_rank,
                        "candidates": raw_candidates,
                    })

    if was_training:
        model.train()

    metrics = {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_examples": total,
        "accuracy_at_1": counts["acc1"] / total,
        "accuracy_at_5": counts["acc5"] / total,
        "accuracy_at_10": counts["acc10"] / total,
        "mean_reciprocal_rank": reciprocal_rank_sum / total,
        "sample": logged_samples[0] if logged_samples else None,
    }

    log_generation_metrics(output_dir, metrics)
    if logged_samples:
        dump_generation_samples(output_dir, tag, logged_samples)
    return metrics


class GenerationAwareSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, generation_eval_config: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_eval_config = generation_eval_config or {}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Standard loss evaluation
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Custom generation evaluation
        gen_cfg = self.generation_eval_config
        if (
            gen_cfg
            and gen_cfg.get("examples")
            and gen_cfg.get("num_generations", 0) > 0
            and gen_cfg.get("tokenizer") is not None
        ):
            tag = metric_key_prefix
            if tag not in {"epoch0", "final"}:
                step = getattr(self.state, "global_step", 0) or 0
                tag = f"{metric_key_prefix}_step{step}"

            gen_metrics = evaluate_with_generations(
                model=self.model,
                tokenizer=gen_cfg["tokenizer"],
                examples=gen_cfg["examples"],
                num_generations=gen_cfg["num_generations"],
                max_input_length=gen_cfg["max_input_length"],
                max_target_length=gen_cfg["max_target_length"],
                batch_size=gen_cfg["batch_size"],
                output_dir=gen_cfg["output_dir"],
                tag=tag,
                num_beams=gen_cfg["num_beams"],
                max_logged_examples=gen_cfg["max_logged_examples"],
            )

            if gen_metrics:
                metrics.update({
                    f"{metric_key_prefix}_accuracy_at_1": gen_metrics["accuracy_at_1"],
                    f"{metric_key_prefix}_accuracy_at_5": gen_metrics["accuracy_at_5"],
                    f"{metric_key_prefix}_mrr": gen_metrics["mean_reciprocal_rank"],
                })

        return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Precision Management
    if args.fp16 and args.bf16:
        raise ValueError("Please enable only one of --fp16 or --bf16")
    if (args.fp16 or args.bf16) and not torch.cuda.is_available():
        LOGGER.warning("Mixed-precision flags ignored because CUDA is not available")
        args.fp16 = False
        args.bf16 = False

    LOGGER.info("Loading datasets...")
    train_dataset, val_dataset = load_all_datasets(args.dataset_dir, use_dictionary=args.use_dictionary)

    # Subset management for debug
    if args.subset and args.subset > 0:
        LOGGER.warning("Training on subset=%d", args.subset)
        train_dataset = train_dataset.select(range(min(args.subset, len(train_dataset))))
        val_cap = max(1, args.subset // 10) if args.subset >= 10 else min(len(val_dataset), 100)
        val_dataset = val_dataset.select(range(min(val_cap, len(val_dataset))))

    # Save the raw list for generation
    val_examples_for_generation = val_dataset.to_list()

    lengths = collect_lengths([train_dataset, val_dataset])
    LOGGER.info("Detected answer lengths: %s", lengths)

    LOGGER.info("Loading tokenizer/model %s", args.model_name)
    # Note: IT5 Efficient uses standard T5/IT5 tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Add special tokens [SL=x] and [EL=x]
    if not args.no_special_tokens:
        ensure_special_tokens(tokenizer, model, build_length_tokens(lengths))
    else:
        LOGGER.info("Skipping special tokens (--no_special_tokens enabled for ablation study)")

    # Intelligent caching based on model name
    safe_model_name = args.model_name.replace("/", "_")
    tokens_suffix = "no_tokens" if args.no_special_tokens else "tokens"
    dict_suffix = "dict" if args.use_dictionary else "nodict"
    cache_base = args.cache_dir / f"{safe_model_name}_{dict_suffix}_{tokens_suffix}_in{args.max_input_length}_out{args.max_target_length}_sub{args.subset}"
    train_cache = cache_base / "train"
    val_cache = cache_base / "val"

    if train_cache.exists() and val_cache.exists():
        LOGGER.info("Reusing cached tokenized dataset under %s", cache_base)
        tokenized_train = load_from_disk(train_cache)
        tokenized_val = load_from_disk(val_cache)
    else:
        LOGGER.info("Tokenizing splits (workers=%d)...", args.preprocessing_workers)
        preprocess_fn = partial(
            preprocess_batch,
            tokenizer=tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            use_special_tokens=not args.no_special_tokens,
        )

        tokenized_train = train_dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=args.preprocessing_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train",
        )
        tokenized_val = val_dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=args.preprocessing_workers,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val",
        )
        cache_base.mkdir(parents=True, exist_ok=True)
        tokenized_train.save_to_disk(train_cache)
        tokenized_val.save_to_disk(val_cache)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_limit,
        predict_with_generate=True,
        generation_max_length=args.max_target_length + 4,
        generation_num_beams=args.num_beams,
        load_best_model_at_end=False,
        resume_from_checkpoint=True,
        report_to=["none"],
        fp16=args.fp16,
        bf16=args.bf16,
    )

    tokenized_train.set_format(type="torch")
    tokenized_val.set_format(type="torch")

    generation_eval_config = {
        "tokenizer": tokenizer,
        "examples": val_examples_for_generation,
        "num_generations": args.num_generations,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "batch_size": args.eval_batch_size,
        "output_dir": args.output_dir,
        "num_beams": args.num_beams,
        "max_logged_examples": args.generation_log_samples,
    }

    trainer = GenerationAwareSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        generation_eval_config=generation_eval_config,
    )
    
    if not args.skip_initial_eval:
        LOGGER.info("Initial evaluation on validation split...")
        try:
            eval_metrics = trainer.evaluate()
            LOGGER.info(f"Epoch 0 eval -> {eval_metrics}")
        except Exception as e:
            LOGGER.error(f"Error during initial eval: {e}")
    else:
        LOGGER.info("Skipping initial evaluation (--skip_initial_eval set)")

    # Check for existing checkpoints
    checkpoints = sorted(args.output_dir.glob("checkpoint-*"))
    resume_checkpoint = None
    if checkpoints:
        resume_checkpoint = str(checkpoints[-1])
        LOGGER.info(f"Found {len(checkpoints)} checkpoint(s). Resuming from: {checkpoints[-1].name}")
    else:
        LOGGER.info("No checkpoints found. Starting training from scratch.")

    LOGGER.info("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    LOGGER.info("Final evaluation on validation split...")
    final_eval_metrics = trainer.evaluate()
    LOGGER.info(f"Post-training eval -> {final_eval_metrics}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()