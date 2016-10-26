import os
import yaml
import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc
from urbansim.utils import yamlio


# Data store
@orca.injectable('store', cache=True)
def hdfstore(settings):
    return pd.HDFStore(
        os.path.join(misc.data_dir(), settings["store"]),
        mode='r')


# Generator function for configs
def register_config_injectable_from_yaml(injectable_name, yaml_file):
    """
    Create orca function for YAML-based config injectables.
    """
    @orca.injectable(injectable_name, cache=True)
    def func():
        with open(os.path.join(misc.configs_dir(), yaml_file)) as f:
            config = yaml.load(f)
            return config
    return func

# Configs
for yaml_file in ["settings.yaml", "model_structure.yaml", "neighborhood_vars.yaml", "yaml_configs.yaml"]:
    injectable_name = yaml_file.split('.')[0]
    register_config_injectable_from_yaml(injectable_name, yaml_file)

# Model structure
ms = orca.get_injectable('model_structure')

# Geographic level of model
template = ms['template']
geography = ms['geography']
geography_id = ms['geography_id']
orca.add_injectable("template", template)
orca.add_injectable("geography", geography)
orca.add_injectable("geography_id", geography_id)
print 'Model template is: %s. Operating on %s with %s.' % (template, geography, geography_id)


# Generator functions

def register_table_from_store(table_name):
    """
    Create orca function for tables from data store.
    """
    if (table_name == geography) & (geography != 'zone'):
        @orca.table('zones', cache=True)
        def func(store):
            return store[table_name]
    else:
        @orca.table(table_name, cache=True)
        def func(store):
            return store[table_name]
    return func

# Tables from data-store
for table in [geography, 'jobs', 'households', 'persons', 'residential_units', 'nodes', 'edges', 'region']:
    register_table_from_store(table)

# Other
@orca.injectable('year')
def year():
    default_year = 2010
    try:
        iter_var = orca.get_injectable('iter_var')
        if iter_var is not None:
            return iter_var
        else:
            return default_year
    except:
        return default_year
        
@orca.injectable("aggregations")
def aggregations(settings):
    if "aggregation_tables" not in settings or \
            settings["aggregation_tables"] is None:
        return []
    return [orca.get_table(tbl) for tbl in settings["aggregation_tables"]]

@orca.injectable("change_sets")
def change_sets():
    change_sets = {}
    return change_sets

orca.add_injectable("track_changes", True)


def step_sequence(pod = None, core_models_only=False):
    yaml_cfg = yamlio.yaml_to_dict(str_or_buffer='./configs/yaml_configs.yaml')
    model_order = ms['model_order']
    supporting_steps = ms['supporting_steps']

    if pod:
        submodels = []
        for yaml_file in yaml_cfg[pod]:
            model_name = yaml_file.split('.')[0]
            submodels.append(model_name)
        model_pod = supporting_steps[pod] + submodels
        return model_pod

    model_sequence = []
    config_sequence = []
    for model in model_order:
        submodels = []
        for yaml_file in yaml_cfg[model]:
            model_name = yaml_file.split('.')[0]
            submodels.append(model_name)
        config_sequence = config_sequence + submodels
        model_pod = supporting_steps[model] + submodels
        model_sequence = model_sequence + model_pod

    if core_models_only:
        return config_sequence

    model_sequence = supporting_steps['pre_processing'] + model_sequence + supporting_steps['post_processing']
    return model_sequence

# Get list of explanatory variables
def get_config(yaml_file):
    with open(os.path.join(misc.configs_dir(), yaml_file)) as f:
        config = yaml.load(f)
        return config

variables = []
model_order = step_sequence(core_models_only=True)
for model in model_order:
    cf_data = get_config(model + '.yaml')
    model_expression = cf_data['model_expression']
    if type(model_expression) == dict:
        variables.extend(model_expression['right_side'])
    else:
        variables.extend(model_expression)
orca.add_injectable('explanatory_variables', variables)

pods = ms['model_order']
orca.add_injectable('pods', pods)

number_of_pods = len(pods)
orca.add_injectable('number_of_pods', number_of_pods)

# Settings
settings = orca.get_injectable('settings')

# Table relationships
orca.broadcast(geography, 'households', cast_index=True, onto_on=geography_id)
orca.broadcast(geography, 'jobs', cast_index=True, onto_on=geography_id)
orca.broadcast('nodes', geography, cast_index=True, onto_on='node_id')
