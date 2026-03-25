import numpy as np
import pandas as pd
import torch

df=pd.read_excel('data/100T-8 Parameter Data Y2025 to Y2026 Feb for AI Forecast 1.xlsx')
raw_nozzle=df['nozzle_temperature'].astype(float)
true=np.load('result/test_targets.npy')
norm=torch.load('result/norm_stats.pt')
y_mean=norm['y_mean'].numpy()
y_std=norm['y_std'].numpy()
print('y_mean:', y_mean[0,6,0], 'y_std:', y_std[0,6,0])
print('test_norm nozzle min/max', true[:,6,0].min(), true[:,6,0].max())
print('test_denorm (with these stats) min/max', (true[:,6,0]*y_std[0,6,0]+y_mean[0,6,0]).min(), (true[:,6,0]*y_std[0,6,0]+y_mean[0,6,0]).max())
print('excel nozzle min/max', raw_nozzle.min(), raw_nozzle.max())
