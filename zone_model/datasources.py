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
    if os.path.exists('./configs/' + yaml_file):
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
    @orca.table(table_name, cache=True)
    def func(store):
        return store[table_name]
    return func

# Tables from data-store
for table in [geography, 'jobs', 'households', 'residential_units', 'non_residential_units']:
    register_table_from_store(table)

# Other
@orca.injectable('year')
def year():
    default_year = ms['base_year']
    try:
        iter_var = orca.get_injectable('iter_var')
        if iter_var is not None:
            return iter_var
        else:
            return default_year
    except:
        return default_year
