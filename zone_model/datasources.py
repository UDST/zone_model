import os
import pandas as pd

import orca
from urbansim.utils import misc
from urbansim.models import MNLDiscreteChoiceModel

import utils


@orca.injectable('store', cache=True)
def hdfstore(settings):
    return pd.HDFStore(
        os.path.join(misc.data_dir(), settings["store"]),
        mode='r')


# Configs
for yaml_file in ["settings.yaml", "model_structure.yaml", "yaml_configs.yaml"]:
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
print 'Model template is: %s. Operating on %s with %s.' % (template, geography, geography_id)


# Tables from data-store
for table in [geography, 'jobs', 'households', 'residential_units', 'non_residential_units']:
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
    except:
        return default_year


# Change-sets
orca.add_injectable("track_changes", False)


# Set up location choice models as top-level injectable.  This can also be done manually on a per-model basis
location_choice_models = {}
model_configs = orca.get_injectable('yaml_configs')
for model_category in model_structure['models']:
    model_category_attribs = model_structure['models'][model_category]
    
    if model_category_attribs['model_type'] == 'location_choice':
        
        supply_variable = model_category_attribs['supply_variable']
        vacant_variable = model_category_attribs['vacant_variable']
        choosers = model_category_attribs['agents_name']
        alternatives = model_category_attribs['alternatives_name']
        
        yaml_configs = model_configs[model_category]
        
        for config in yaml_configs:
            model_name = config.split('.')[0]
            config_path = misc.config(config)
            model = MNLDiscreteChoiceModel.from_yaml(str_or_buffer=config_path)
            
            model.name = model_name
            model.supply_variable = supply_variable
            model.vacant_variable = vacant_variable
            model.choosers = choosers
            model.alternatives = alternatives
            
            location_choice_models[model_name] = model

orca.add_injectable('location_choice_models', location_choice_models)
