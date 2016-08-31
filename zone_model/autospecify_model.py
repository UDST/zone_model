import orca
orca.add_injectable('process_has_pandana', True)

import datasources
import variables
import models

from autospec import fit_category_constraints

orca.add_injectable('autospec_multiprocess', 'joblib')

orca.run(['build_networks', 'train_model'])
