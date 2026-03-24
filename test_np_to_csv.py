import numpy as np
import pandas as pd

pred_path = "/scratch/prj0000000262/Sanwa_forecast/Sanwa_forecast_ft/checkpoints/FT_tiny_Sanwa_forecast_from1_Sw_fc_bs8_lr3e-06_seed42_20260113_144819/test_preds.npy"
tgt_path  = "/scratch/prj0000000262/Sanwa_forecast/Sanwa_forecast_ft/checkpoints/FT_tiny_Sanwa_forecast_from1_Sw_fc_bs8_lr3e-06_seed42_20260113_144819/test_targets.npy"

pred = np.load(pred_path)
tgt  = np.load(tgt_path)

df = pd.concat([
    pd.DataFrame(pred, columns=[f"pred_{i}" for i in range(pred.shape[1])]),
    pd.DataFrame(tgt,  columns=[f"target_{i}" for i in range(tgt.shape[1])]),
], axis=1)

out_csv = pred_path.replace("test_preds.npy", "test_preds_targets_combined.csv")
df.to_csv(out_csv, index=False)

print("saved:", out_csv, "shape:", df.shape)
