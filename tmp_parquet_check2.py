import pyarrow.parquet as pq, numpy as np
from pathlib import Path
base = Path(__file__).parent
out = base / 'tmp_parquet_check_output2.txt'
with out.open('w') as f:
    for split in ['train', 'val', 'test']:
        p = base / 'splits_h50_chrono_option2' / f'{split}.parquet'
        if not p.exists():
            f.write(f'{split} missing\n')
            continue
        pf = pq.ParquetFile(p)
        f.write(f'{split} rows={pf.metadata.num_rows} groupr={pf.num_row_groups}\n')
        tab = pf.read_row_group(0)
        if 'labels' not in tab.column_names:
            f.write('no labels\n')
            continue
        labels = tab['labels'].combine_chunks()
        y = labels.values.to_numpy(zero_copy_only=False).astype(np.float32)
        H=50; C=11
        y_mat = y.reshape(-1,H,C)
        ch6=y_mat[:,:,6]
        f.write(f'ch6 min {float(ch6.min())} max {float(ch6.max())} mean {float(ch6.mean())}\n')
        means = y_mat.mean(axis=(0,1)).tolist()
        f.write('means '+','.join(str(m) for m in means)+'\n')
        f.write('ch6 first10 '+','.join(str(float(x)) for x in ch6[:10].flatten())+'\n')

print('done')
