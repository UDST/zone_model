import orca
orca.add_injectable('process_has_pandana', True)

import datasources
import variables
import models

from autocalib import brute_force

mpo_id = int(orca.get_injectable('store')['region']['mpo_id'].values[0])

orca.add_injectable('mpo_id', mpo_id)

orca.run(['run_calib'])
