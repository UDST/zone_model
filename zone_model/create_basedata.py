import os
from pipeline_io import to_s3
from block_model_data import mpo

#mpo_id = 27197000
# MTC 6197001
# St. Cloud Area Planning Organization (APO), MN, St. Cloud, 363 sqmi, 130191 pop, 1970 yr founded.  27197000 
#Corridor Metropolitan Planning Organization  19196400
#Des Moines Area MPO  19198300  
#Pikes Peak Area COG   8197702


#output_dir = os.path.expanduser('~/modeldata/mpo%s' % mpo_id)

#from block_model_data.base_year_data import create_model_h5
#create_model_h5(output_dir)

#mpo.get_data_for_region(mpo_id, output_dir)

#to_s3(os.path.join(output_dir, 'model_data.h5'), '/modeldata/mpo%s/model_data.h5' % mpo_id)


"""
for mpo_id in [8197702]: #19198300, 27197000, 19196400, 8197702
    output_dir = os.path.expanduser('~/modeldata/mpo%s' % mpo_id)
    mpo.get_data_for_region(mpo_id, output_dir)
    to_s3(os.path.join(output_dir, 'model_data.h5'), '/modeldata/mpo%s/model_data.h5' % mpo_id)
"""


#to_s3('./data/scen1_run1.h5', '/modeldata/mpo%s/runs/scen1_run1.h5' % mpo_id)

import pandas as pd
import numpy as np
import shutil
import time

coords_df = pd.read_csv('http://synthpop-data2.s3-website-us-west-1.amazonaws.com/us_mpo_data_2015_v2.csv')
coords_df = coords_df[~coords_df.STATE.isin(['TX', 'MA', 'AK'])]

for mpo_id in coords_df.MPO_ID.values[:100]:
#for mpo_id in [27197000]:
    mpo_id = int(mpo_id)
    print mpo_id
    output_dir = os.path.expanduser('~/modeldata/mpo%s' % mpo_id)
    try:
        result = mpo.get_data_for_region(mpo_id, output_dir)
        if result:
            to_s3(os.path.join(output_dir, 'model_data.h5'), '/modeldata/mpo%s/model_data.h5' % mpo_id)
            print 'About to delete!'
            time.sleep(1)
            shutil.rmtree(output_dir)
    except:
        print 'Data generation failed for MPO %s.' % mpo_id

