import os
import pandas as pd
import string
import ast
import argparse


def process_prediction(pred_path, test_df, output_path):
    """
    Load, format and save a single prediction.
    """
    pred_df = pd.read_csv(pred_path)
    pred_df["clue"] = test_df["clue"]
    pred_df["answer_length"] = test_df["answer_length"]

    # Extract the first candidate as answer, use 'dummya' if empty
    def get_first_candidate(candidates):
        if pd.notnull(candidates) and candidates != "":
            return candidates.split(";")[0].strip(" '\"")
        return "dummya"

    pred_df["answer"] = pred_df["candidates"].apply(get_first_candidate)
    pred_df["candidates"] = pred_df["candidates"].apply(
        lambda x: x.split(";") if pd.notnull(x) and x != "" else []
    )
    pred_df[["clue", "answer", "answer_length", "candidates"]].to_csv(
        output_path, index=False
    )

    # For constrained, pad candidates to 10 (keeping empty slots)
    if "constrained" in output_path:
        df = pd.read_csv(output_path)
        for idx, row in df.iterrows():
            candidates = row["candidates"]
            # Keep empty slots as they are
            if pd.notnull(candidates) and candidates != "":
                cand_list = ast.literal_eval(candidates)
                if len(cand_list) < 10:
                    num_dummies = 10 - len(cand_list)
                    dummies = [
                        f"dummy{string.ascii_lowercase[len(cand_list) + k]}"
                        for k in range(num_dummies)
                    ]
                    cand_list.extend(dummies)
                    df.at[idx, "candidates"] = str(cand_list)
        df.to_csv(output_path, index=False)


def main(test_path, output_dir, constrained_pred, unconstrained_pred):
    """
    Main function to process predictions.

    Args:
        test_path: Path to test dataset CSV
        output_dir: Directory to save formatted predictions
        constrained_pred: Path to constrained predictions CSV
        unconstrained_pred: Path to unconstrained predictions CSV
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load Test set
    test_df = pd.read_csv(test_path)

    # Configurations for predictions
    configs = []

    if constrained_pred:
        configs.append(
            {
                "pred_path": constrained_pred,
                "output_path": os.path.join(
                    output_dir, "final_constrained_predictions.csv"
                ),
            }
        )

    if unconstrained_pred:
        configs.append(
            {
                "pred_path": unconstrained_pred,
                "output_path": os.path.join(
                    output_dir, "final_unconstrained_predictions.csv"
                ),
            }
        )

    # Process each configuration
    for config in configs:
        process_prediction(config["pred_path"], test_df, config["output_path"])

    # Check in constrained if there are at least 10 candidates for each row
    if constrained_pred:
        constrained_output = os.path.join(
            output_dir, "final_constrained_predictions.csv"
        )
        print(f"Checking candidates in {constrained_output}...")
        df = pd.read_csv(constrained_output)
        insufficient_candidates = df["candidates"].apply(
            lambda x: len(ast.literal_eval(x)) < 10
            if pd.notnull(x) and x != ""
            else True
        )
        problematic_rows = insufficient_candidates[
            insufficient_candidates
        ].index.tolist()
        if problematic_rows:
            print(f"Found {len(problematic_rows)} rows with less than 10 candidates.")
            print(f"Rows with less than 10 candidates at rows: {problematic_rows}")
            for pr in problematic_rows:
                print(
                    f"Row {pr}: {df.at[pr, 'clue']}; candidates: '{df.at[pr, 'candidates']}'"
                )
        print("Check completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format Task 1 predictions for submission"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="hf_evalita2026\\task_1\\datasets\\test_gold.csv",
        help="Path to test dataset CSV (default: hf_evalita2026/task_1/datasets/test_gold.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="RESULT_FOR_SUBMISSION/task1_runset",
        help="Output directory for formatted predictions (default: RESULT_FOR_SUBMISSION/task1_runset)",
    )
    parser.add_argument(
        "--constrained",
        type=str,
        default="results/task1_constrained_checkpoint-23epochs.csv",
        help="Path to constrained predictions CSV (default: results/task1_constrained_checkpoint-23epochs.csv)",
    )
    parser.add_argument(
        "--unconstrained",
        type=str,
        default="results/task1_unconstrained_checkpoint-23epochs.csv",
        help="Path to unconstrained predictions CSV (default: results/task1_unconstrained_checkpoint-23epochs.csv)",
    )

    args = parser.parse_args()
    main(args.test, args.output_dir, args.constrained, args.unconstrained)
