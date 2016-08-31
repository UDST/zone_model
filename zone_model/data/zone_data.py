import pandas as pd
store = pd.HDFStore('model_data.h5')

tbls = {}

for tbl in store.keys():
    df = store[tbl]
    tbls[tbl] = df


for tbl in tbls.items():
    tbl_name = tbl[0]
    df = tbl[1]
    if df.index.name == 'block_id':
        tbls[tbl_name].index.name = 'zone_id'
    if 'block_id' in df.columns:
        tbls[tbl_name] = df.rename(columns = {'block_id' : 'zone_id'})

for tbl in tbls.items():
    tbl_name = tbl[0]
    df = tbl[1]
    store[tbl_name] = df

store.close()
