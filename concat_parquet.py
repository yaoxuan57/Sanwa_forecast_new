import pandas as pd
from pathlib import Path

folder = Path("windows_out")
files = sorted(folder.glob("*_dataset.parquet"))  # keeps 12_17, 12_19, 12_22 order

df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

df.to_parquet(folder / "12_17_12_19_12_22_concat.parquet", index=False)

print(df.shape)
