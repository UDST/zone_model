import yaml
import time
import numpy as np
import pandas as pd

import orca
import urbansim
from urbansim.utils import yamlio

import autospec_recipes
from autospec.autospec_choice import estimation_function

orca.add_injectable('process_has_pandana', True)

import datasources
import variables
import models

template = orca.get_injectable('template')
geography_base_id = orca.get_injectable('geography_id')

orca.run(['build_networks'])

specifications = {}
dcms = {}
model_yaml_filenames = []
max_iter = 0

model_name = 'hlcm'
alternatives_name = 'zones'
alternative_id_name = geography_base_id
agents_name = 'households'

constraint_config = orca.get_injectable('constraint_configs')[template]['hlcm_constraints.yaml']

alternatives = orca.get_table(alternatives_name)
explanatory_variables = [var for vars in [category['variables']  for category in constraint_config] for var in vars]
alternatives = alternatives.to_frame(alternatives.local_columns + explanatory_variables)

choosers = orca.get_table(agents_name).to_frame()

alts = alternatives
dtypes = alts.dtypes.reset_index().rename(columns={'index':'col', 0:'dtype'})
numeric_cols = [alt[0] for alt in zip(dtypes.col, dtypes.dtype) if alt[1] != 'object'] # Only keep numeric columns
alts = alts[numeric_cols]

agents_for_estimation = choosers
agents_for_estimation = agents_for_estimation[agents_for_estimation['recent_mover'] == 1]

model_estim_fn = estimation_function(agents_for_estimation, alts, alternative_id_name,
                                            choosers_fit_filter='recent_mover == 1')

base_specification = set([])
variable_pool = set(alts.columns)

# from autospec.autospec_choice import evaluate_variable_set

# log_likelihood_ratios, t_scores, models = evaluate_variable_set(variable_pool,
#                                                         base_specification,
#                                                         model_estim_fn)

## SEQUENTIAL

start_time = time.time()

for variable in variable_pool:
    specification_proposal = set(base_specification)
    specification_proposal.add(variable)
    print 'Testing %s in specification' % variable
    try:
        dcm = model_estim_fn(specification_proposal)
        result = t_score = dcm.fit_parameters['T-Score'].loc[variable]
    except:
        print 'Estimation problem when estimating with %s.  E.g. singular matrix.' % variable
        continue
        
end_time = time.time()
time_elapsed1 = end_time - start_time
print 'Duration: %s minutes' % (time_elapsed1/60)


def estimate_model(variable):
    specification_proposal = set(base_specification)
    specification_proposal.add(variable)
    try:
        dcm = model_estim_fn(specification_proposal)
        result = t_score = dcm.fit_parameters['T-Score'].loc[variable]
    except:
        result = -999
    return result


time.sleep(3)
#### JOBLIB

from joblib import Parallel, delayed

start_time = time.time()
results = Parallel(n_jobs=3)(delayed(estimate_model)(variable) for variable in variable_pool)
end_time = time.time()
time_elapsed2 = end_time - start_time
print 'Duration: %s minutes' % (time_elapsed2/60)


time.sleep(3)
#### DASK DISTRIBUTED

from distributed import Executor, LocalCluster # must install from source

c = LocalCluster(n_workers = 3)
executor = Executor(c)

start_time = time.time()

a = executor.map(estimate_model, variable_pool)

results = executor.gather(a)

end_time = time.time()
time_elapsed3 = end_time - start_time
print 'Duration: %s minutes' % (time_elapsed3/60)



####

print 'sequential', time_elapsed1
print 'joblib', time_elapsed2
print 'dask_distr', time_elapsed3


"""
from pymongo import MongoClient

client = MongoClient()

db = client.test

def dcm_to_dict(dcm):
    model = dcm.to_dict()
    model['fit_parameters']['Std Error'] = model['fit_parameters'].pop('Std. Error')
    model['model_expression'] = list(model['model_expression'])
    return model

spec_proposals = [dcm_to_dict(dcm) for dcm in dcms]


result = db.specs.insert_many(spec_proposals)

cursor = db.specs.find()
for document in cursor:
    print(document)
"""

