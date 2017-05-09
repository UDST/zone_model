from __future__ import print_function

import os
import pandas as pd

import orca
from urbansim.utils import misc

from zone_model import utils


@orca.injectable('store', cache=True)
def hdfstore(settings):
    return pd.HDFStore(
        os.path.join(misc.data_dir(), settings["store"]),
        mode='r')


# Configs
for yaml_file in ["settings.yaml", "model_structure.yaml",
                  "yaml_configs.yaml"]:
    if os.path.exists('./configs/' + yaml_file):
        injectable_name = yaml_file.split('.')[0]
        utils.register_config_injectable_from_yaml(injectable_name, yaml_file)

# Geographic level of model
model_structure = orca.get_injectable('model_structure')
template = model_structure['template']
geography = model_structure['geography']
geography_id = model_structure['geography_id']
orca.add_injectable("template", template)
orca.add_injectable("geography", geography)
orca.add_injectable("geography_id", geography_id)
print('Model template is: {}. Operating on {} with {}.'
      .format(template, geography, geography_id))

# Tables from data-store
for table in [geography, 'jobs', 'households', 'residential_units',
              'non_residential_units']:
    utils.register_table_from_store(table)


@orca.injectable('year')
def year():
    default_year = model_structure['base_year']
    try:
        iter_var = orca.get_injectable('iter_var')
        if iter_var is not None:
            return iter_var
        else:
            return default_year
    except KeyError:
        return default_year


# Change-sets
orca.add_injectable("track_changes", False)

# Set up location choice model objects.
# Register as injectable to be used throughout simulation
location_choice_models = {}
model_configs = utils.get_model_category_configs()
for model_category_name, model_category_attributes in model_configs.items():
    if model_category_attributes['model_type'] == 'location_choice':
        model_config_files = model_category_attributes['config_filenames']

        for model_config in model_config_files:
            model = utils.create_lcm_from_config(model_config,
                                                 model_category_attributes)
            location_choice_models[model.name] = model

orca.add_injectable('location_choice_models', location_choice_models)


@orca.injectable("change_sets")
def change_sets():
    change_sets = {}
    return change_sets

orca.add_injectable("track_changes", True)
