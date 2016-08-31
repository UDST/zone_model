import os
import sys
import time
import redis
import subprocess

import orca
import datasources

python = sys.executable
root_path = os.path.dirname(__file__)


def run(filename, args=None):
    """"Run Python file relative to script without blocking."""
    path = os.path.join(root_path, filename)
    command = [python, path]
    if args:
        command = command + args
    return subprocess.Popen(command)


def check_run(filename):
    """Run Python file relative to script, block, assert exit code is zero."""
    path = os.path.join(root_path, filename)
    return subprocess.check_call([python, path])


first_year = 2011
forecast_year = 2012

r = orca.get_injectable('redis_conn')
r.set('first_year', first_year)
r.set('forecast_year', forecast_year)
r.set('start_time', str(time.time()))
r.set('simulation_status', 'started')

model_pods = orca.get_injectable('pods') # already in order

model_time_interval = .1
#run('simulate.py', args = ['-r', '.02', '-y', str(forecast_year)])

#Launch the sub-processes (for network service, and then each of the model pods in order)
run('network_service.py')
time.sleep(5)
for pod in model_pods:
    run('sim_parallel.py', args = ['-p',  pod, '-y', str(forecast_year)]) # '-c'
    time.sleep(model_time_interval)

# Wait until simulation finishes, then log status and exit
while True:
    time.sleep(.5)
    sim_status = r.get('simulation_status')
    if sim_status == 'Done':
        break
print 'Simulation process manager exiting'
time.sleep(1)


# Cleanup for calibration (make this an arg)
#for f in os.listdir('./data'):
#    if f.startswith('change_set'):
#        os.remove('./data/' + f)

