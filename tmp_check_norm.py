import numpy as np
import torch
from pathlib import Path

base = Path(__file__).parent
stats = torch.load(base / 'result' / 'norm_stats.pt')
print('y_mode', stats.get('y_mode'), 'x_mode', stats.get('x_mode'))
print('y_mean', stats['y_mean'].reshape(-1).tolist())
print('y_std', stats['y_std'].reshape(-1).tolist())

target = np.load(base / 'result' / 'test_targets.npy')
print('target shape', target.shape)
print('target nozzle index 6, h0 norm min/max/mean', float(target[:,6,0].min()), float(target[:,6,0].max()), float(target[:,6,0].mean()))
print('target nozzle index 6, h0 first 10 delicious', (target[:10,6,0]*stats['y_std'][0,6,0]+stats['y_mean'][0,6,0]).tolist())
