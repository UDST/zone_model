import orca
from urbansim.utils import misc

import utils
import datasources
import variables


# Transition
utils.register_simple_transition_model('jobs', .005)
utils.register_simple_transition_model('households', .005)
utils.register_simple_transition_model('residential_units', .005)
utils.register_simple_transition_model('non_residential_units', .005)

    
'''
try:
    from google.cloud import datastore
    client = datastore.Client()
except:
    pass

def get_selected_models_by_version(model_version_id):
    query = client.query(kind='Spec')
    query.add_filter('run_id', '=', model_version_id)
    query.add_filter('selected', '=', True)
    return list(query.fetch())

models_version_id = '70320150328dnh'  ## TODO: pull this into config file
models = get_selected_models_by_version(models_version_id)
'''

