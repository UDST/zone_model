import orca

from zone_model import datasources
from zone_model import variables
from zone_model import models

transition_models = ['simple_jobs_transition',
                     'simple_households_transition',
                     'simple_residential_units_transition',
                     'simple_non_residential_units_transition']

choice_models = ['elcm1',
                 'hlcm1', 'hlcm2', 'hlcm3', 'hlcm4',
                 'rdplcm1',
                 'nrdplcm1']

orca.run(transition_models + choice_models)
