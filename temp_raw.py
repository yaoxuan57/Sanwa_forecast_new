import pandas as pd
raw_path='data/100T-8 Parameter Data Y2025 to Y2026 Feb for AI Forecast 1.xlsx'
df=pd.read_excel(raw_path)
print('raw head nozzle', df['nozzle_temperature'].head(20).to_list())
print('raw tail nozzle', df['nozzle_temperature'].tail(20).to_list())
print('raw overall min/max/mean', df['nozzle_temperature'].min(), df['nozzle_temperature'].max(), df['nozzle_temperature'].mean())
