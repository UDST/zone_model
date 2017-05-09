import orca
import numpy as np

from zone_model import utils
from zone_model import datasources
from zone_model import variables


# Transition
utils.register_simple_transition_model('jobs', .005)
utils.register_simple_transition_model('households', .005)
utils.register_simple_transition_model('residential_units', .005)
utils.register_simple_transition_model('non_residential_units', .005)

# Relocation
utils.register_simple_relocation_model('jobs', .01)
utils.register_simple_relocation_model('households', .01)

# Location Choice
location_choice_models = orca.get_injectable('location_choice_models')
for name, model in location_choice_models.items():
    utils.register_choice_model_step(model.name,
                                     model.choosers,
                                     choice_function=utils.unit_choices)


# Ensemble example- soft voting
@orca.step('simple_ensemble')
def simple_hlcm_ensemble(households, location_choice_models):
    model_names = ['hlcm1', 'hlcm2', 'hlcm3', 'hlcm4']
    hlcm_models = [location_choice_models[model] for model in model_names]
    hlcm_weights = [.25, .25, .25, .25]

    model = utils.SimpleEnsemble(hlcm_models, hlcm_weights)
    choices = model.simulate(choice_function=utils.random_choices)

    households.update_col_from_series('zone_id', choices)


'''
try:
    from google.cloud import datastore
    client = datastore.Client()
except:
    pass

def get_selected_models_by_version(model_version_id):
    """
    Get model configs from Cloud Datastore.
    """
    query = client.query(kind='Spec')
    query.add_filter('run_id', '=', model_version_id)
    query.add_filter('selected', '=', True)
    return list(query.fetch())

models_version_id = '70320150328dnh'  ## TODO: pull this into config file
models = get_selected_models_by_version(models_version_id)
'''
