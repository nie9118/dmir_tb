import torch.optim as optim
from torch.utils.data import DataLoader
from trm.config import TrainingConfig
from trm.training import train
from trm.model.trm import TinyRecursiveModel
from trm.data.sudoku import SudokuDataset
from trm.data.utils import create_train_test_datasets
from trm.model.utils import EMA


def main():
    cfg = TrainingConfig()

    print(f"Using device: {cfg.device}")

    dataset = SudokuDataset(num_samples=2000, difficulty=0.5)
    train_dataset, test_dataset = create_train_test_datasets(dataset, train_ratio=0.95, random_seed=42)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = TinyRecursiveModel(
        dim=cfg.dim,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        n_layers=cfg.n_layers,
        n_recursions=cfg.n_recursions,
        T_recursions=cfg.t_recursions,
        use_attention=cfg.use_attention,
    ).to(cfg.device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ema = EMA(model, decay=cfg.ema_decay)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    print("\n--- Starting Training ---")
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
    train(model, train_loader, eval_loader, optimizer, ema, cfg)
    print("\n--- Training Finished ---")


if __name__ == '__main__':
    main()