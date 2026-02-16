# FFT-UniBa at Cruciverb-IT: Special Length Tokens and CSP for Italian Crossword Solving

**Authors**: Andrea Porcelli, Filippo di Gravina, Emanuele Fontana, Mattia Curri, Francesco Damiano di Gregorio

This repository contains the code for the FFT-UniBa submission to the Cruciverb-IT shared task at EVALITA 2026.

## Preliminaries

Before running the code, ensure you have the required packages installed. You can do this by using `uv`:

```bash
uv sync
```

### Finetuning the IT5-Efficient Model

Download the dataset from [here](https://huggingface.co/datasets/cruciverb-it/evalita2026) and place it in the `data/task_1/datasets` directory.

To finetune the IT5-Efficient model on the Cruciverb-IT dataset with special tokens and dictionary augmentation, use the following command:

```bash
uv run scripts/it5eff_official.py --use_dictionary -skip_initial_eval --model_name data/task_1/datasets --output_dir <OUTPUTDIR>
```

**Parameters:**
- `--use_dictionary`: Enables dictionary augmentation (downloaded automatically from Hugging Face)
- `-skip_initial_eval`: Skips the initial evaluation step
- `--model_name data/task_1/datasets`: Path to the downloaded dataset (fixed)
- `<OUTPUTDIR>`: **Replace with your desired output directory** where the finetuned model will be saved (e.g., `models/finetuned_model`)

The dictionary augmentation is enabled with the `--use_dictionary` flag, it will be downloaded and loaded automatically from Hugging Face. The hyperparams used in the paper are set as default in the script, you can modify them as needed.

To evaluate the results, run:

```bash
uv run .\scripts\generate_task1_dual_mode.py --model_path <OUTPUTDIR_EVAL> --input_file data/task_1/datasets/test.csv --output_dir <OUTPUTDIR_EVAL>

uv run .\scripts\task1_formatter.py --output-dir <OUTPUTDIR_FORMAT> --constrained <PATH/final_constrained_predictions.csv> --unconstrained <PATH/final_unconstrained_predictions.csv>
```

**Parameters for `generate_task1_dual_mode.py`:**
- `--model_path <OUTPUTDIR_EVAL>`: **Replace with the path to your finetuned model** (same as `<OUTPUTDIR>` from the finetuning step)
- `--input_file data/task_1/datasets/test.csv`: Path to the test dataset (fixed)
- `--output_dir <OUTPUTDIR_EVAL>`: **Replace with your desired output directory** where predictions will be saved

**Parameters for `task1_formatter.py`:**
- `--output-dir <OUTPUTDIR_FORMAT>`: **Replace with your desired output directory** for formatted results
- `--constrained <PATH/final_constrained_predictions.csv>`: **Replace with the path** to the constrained predictions from the previous step
- `--unconstrained <PATH/final_unconstrained_predictions.csv>`: **Replace with the path** to the unconstrained predictions from the previous step

Then, you can use the provided evaluation script from the shared task to evaluate your predictions.

### Generate Candidates for Sub-task 2

To generate candidates for the crossword clues using the finetuned model, run:

```bash
uv run .\scripts\generate_task2_dual_mode.py --model_path <OUTPUTDIR> --output_dir <OUTPUTDIR_TASK2> --num_return_sequences 1000 --num_beams 600
```

**Parameters:**
- `--model_path <OUTPUTDIR>`: **Replace with the path to your finetuned model** (same as from the finetuning step)
- `--output_dir <OUTPUTDIR_TASK2>`: **Replace with your desired output directory** where the generated candidates will be saved
- `--num_return_sequences 1000`: Number of sequences to generate for each clue (fixed)
- `--num_beams 600`: Beam search width for generation (fixed)

### Run the Sub-Task 2 algorithm

To solve the crosswords using the generated candidates, run:

```bash
uv run scripts\task_2\task2_algorithm.py --candidates <OUTPUTDIR_TASK2> --output_file <OUTPUTDIR_TASK2_DEBUG> --output_grid_file <OUTPUTDIR_TASK2_GRIDS>
```

**Parameters:**
- `--candidates <OUTPUTDIR_TASK2>`: **Replace with the path** to the candidates generated in the previous step
- `--output_file <OUTPUTDIR_TASK2_DEBUG>`: **Replace with your desired output file path** for debug/detailed information
- `--output_grid_file <OUTPUTDIR_TASK2_GRIDS>`: **Replace with your desired output file path** for the solved crossword grids

Then you can use the provided evaluation script from the shared task to evaluate your crossword solutions.


## Citation

If you find our work useful to your research and applications, please consider citing the paper and staring the repo ☺️.

**BibTeX:**

```bibtex
TBD
```


---

<div align="center">

🚀

</div>