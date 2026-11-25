from pathlib import Path

root = Path(r"C:\Users\ACER\OneDrive\Desktop\HTR_Project")
checkpoints = root / "checkpoints"

for model in ["baseline", "bilstm", "attention", "attention_aug"]:
    print(f"\n{model.upper()}")
    for seed in range(3):
        path = checkpoints / f"{model}_seed{seed}.pt"
        status = "FOUND" if path.exists() else "MISSING"
        size = f"{path.stat().st_size / (1024**2):.1f} MB" if path.exists() else ""
        print(f"  Seed {seed}: {status} {size}")