import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
base = Path(__file__).parent
path = base / 'splits_h50_chrono_option2' / 'test.parquet'
pf = pq.ParquetFile(path)
print('rows', pf.metadata.num_rows, 'row_groups', pf.num_row_groups)
# read first batch of labels
batch = pf.read_row_group(0, columns=['labels'])
labels_col = batch.column(batch.schema.get_field_index('labels')).combine_chunks()
y_vals = labels_col.values.to_numpy(zero_copy_only=False).astype(np.float32)
print('labels flat shape', y_vals.shape, 'list_size', labels_col.type.list_size)
# build first sample as H,C to inspect range and mean
h=50
c=11
first = y_vals[:h*c].reshape(h,c)
print('first sample per-channel min,max,mean')
for i in range(c):
    arr=first[:,i]
    print(i, arr.min(), arr.max(), arr.mean())
# compute overall stats across entire set maybe huge; sample first 1000 values
print('first 10 nozzle values', first[:10,6])
