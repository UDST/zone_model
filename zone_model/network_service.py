import os
import time
import yaml
import redis

import orca
orca.add_injectable('process_has_pandana', True)
orca.add_injectable('pod', 'network')

import pandas as pd
from urbansim.utils import misc

import datasources
import variables
import models

r = orca.get_injectable('redis_conn')
r.set('pod_count', 0)
year = int(r.get('first_year'))
r.set('year', year)
orca.add_injectable('year', year)

number_of_pods = orca.get_injectable('number_of_pods')


variables = orca.get_injectable('explanatory_variables')
orca.set_redis_columns(variables)

access_vars = [var for var in variables if var.endswith('m') | ('agg' in var) | ('00m' in var) | ('km' in var)] #make this more intelligent
other_vars = [var for var in variables if var not in access_vars]  #todo:  ensure that only unique entries

# Setting initial state of variables
for access_var_name in access_vars:
    r.set(access_var_name, 'Not calculated')
for other_var_name in other_vars:
    r.set(other_var_name, 'Not calculated')

# Pandana set up
# Other pre-processing steps should go here too
orca.run(['build_networks'])

# Variable calculations
def variable_calcs():
    print 'Starting calculation of network variables'
    for access_var_name in access_vars:
        access_var = orca.get_table('zones')[access_var_name]
    print 'Network variable calculations done.'

    print 'Starting calculation of other non-network variables'
    for other_var_name in other_vars[::-1]:  # In reverse order
        other_var = orca.get_table('zones')[other_var_name]
    print 'Other non-network variable calculations done'

# inter-process redis check in
def model_pod_checkin():
    print 'Waiting for pods to finish...'
    while True:
        time.sleep(.1)
        pod_count = r.get('pod_count')
        if pod_count == str(number_of_pods):
            print 'All pods have finished. Resetting state of variable keys in redis as uncalculated'
            break

def wrap_up_year(year):
    # Incorporate change sets
    orca.run(['incorporate_change_sets'])

    for access_var_name in access_vars:
        r.set(access_var_name, 'Not calculated')
    for other_var_name in other_vars:
        r.set(other_var_name, 'Not calculated')

    r.incr('year')
    year = year + 1

    orca.clear_cache()
    orca.add_injectable('year', year)
    orca.add_injectable('process_has_pandana', True)
    orca.add_injectable('pod', 'network')

    # Reset pod count
    r.set('pod_count', 0)

    print 'Ready for the next simulation year'
    return year

def manage_year(year):
    variable_calcs()
    model_pod_checkin()
    next_year = wrap_up_year(year)
    print next_year
    return next_year

forecast_year = int(r.get('forecast_year'))
print 'Forecast year is %s' % forecast_year
# for year_iter in range(year, forecast_year + 1):
while year <= forecast_year:
    year = manage_year(year)

print 'Done!'

#  Add in post-processing models here
orca.run(['calibration_indicators'])

end_time = time.time()
r.set('end_time', str(end_time))
start_time = float(r.get('start_time'))

time_elapsed = end_time - start_time
print 'Simulation duration: %s minutes' % (time_elapsed/60)
r.set('simulation_status', 'Done')
