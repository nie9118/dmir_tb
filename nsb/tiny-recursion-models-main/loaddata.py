from datasets import load_dataset
dataset = load_dataset("sapientinc/sudoku-extreme-1k", split="train")
dataset.save_to_disk("path/to/local/dataset")