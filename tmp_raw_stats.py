import pandas as pd
from pathlib import Path
base = Path(__file__).parent
raw_path = base / 'data' / '100T-8 Parameter Data Y2025 to Y2026 Feb for AI Forecast 1.xlsx'
if not raw_path.exists():
    print('missing raw file', raw_path)
else:
    df = pd.read_excel(raw_path)
    cols=['injection_time','switchover_position','max_injection_pressure','switchover_pressure','end_of_packing_stroke','plastification_time','nozzle_temperature','barrel_front_temperature','barrel_center_temperature','barrel_rear_temperature','feeder_temperature']
    for c in cols:
        if c in df.columns:
            print(c, 'min/max/mean', df[c].min(), df[c].max(), df[c].mean())
        else:
            print('missing', c)
