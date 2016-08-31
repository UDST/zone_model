import os
import sys
import argparse
import numpy as np

import orca
from autospec import sim_utils
from urbansim.utils import yamlio
from urbansim.utils import misc

import datasources

variables = orca.get_injectable('explanatory_variables')
orca.set_redis_columns(variables)

orca.add_injectable('tracking', False)

def run(forecast_year=2011, growth_rate=.01, random_seed=0, calibrated=False, data_out=False, pod=False):
    """
    Set up and run simulation.

    Parameters
    ----------
    forecast_year : int, optional
        Year to simulate to.
    growth_rate : float, optional
        Assumed growth-rate over forecast period (value of .01 means a 1% rate)
    random_seed : int, optional
        Random seed.
    calibrated : boolean, optional
        Indicates whether to use calibrated coefficients.  If True, requires calib_dummies.csv in data dir.
    data_out : boolean, optional
        Indicates whether to write output to HDF5.
    pod : False or str
        Model pod name to run if only single pod (e.g job models) is desired, optional
    Returns
    -------
    _ : None
        No return value for now.
    """

    # Set value of random seed
    np.random.seed(random_seed)

    # Optionally use calibrated coefficients in simulation
    if calibrated:
        orca.add_injectable('calibrated', True)

    # Set model parameters
    orca.add_injectable('growth_rate', growth_rate)

    # Indicates whether the current pyton process is model or worker, if running distributed
    if orca.get_injectable('multiprocess') == False:
        orca.add_injectable('process_has_pandana', True)
    else:
        orca.add_injectable('process_has_pandana', False)

    # Register models, variables
    import variables
    import models

    # Register auto-fitted models with orca
    yaml_cfg = yamlio.yaml_to_dict(str_or_buffer='./configs/yaml_configs.yaml')
    hlcm = sim_utils.register_orca_steps_for_segmented_model(yaml_cfg['hlcm'], models.make_hlcm_func)
    elcm = sim_utils.register_orca_steps_for_segmented_model(yaml_cfg['elcm'], models.make_elcm_func)
    rdplcm = sim_utils.register_orca_steps_for_segmented_model(yaml_cfg['rdplcm'], models.make_rdplcm_func)
    rent_repm = sim_utils.register_orca_steps_for_segmented_model(yaml_cfg['repm_rent'], models.make_repm_func)
    value_repm = sim_utils.register_orca_steps_for_segmented_model(yaml_cfg['repm_value'], models.make_repm_func)

    # Simulate
    if pod != False:
        orca.add_injectable('pod', pod)
    coordination_steps = ['export_change_sets', 'redis_checkin1', 'incorporate_change_sets', 'redis_checkin2']
    step_sequence = datasources.step_sequence() if (pod == False) else datasources.step_sequence(pod) + coordination_steps
    
    import traceback
    try:
        if data_out:
            orca.run(step_sequence, iter_vars = range(2011, forecast_year + 1), data_out='./data/test.h5')
        else:
            orca.run(step_sequence, iter_vars = range(2011, forecast_year + 1))
    except Exception, e:
        traceback.print_exc(file=open("errlog.txt","a"))
        #with open("debug.txt", "w") as text_file:
        #    text_file.write(str(e))


if __name__ == '__main__':

    # Run simulation with optional command-line arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-y", "--year", type=int, help="forecast year to simulate to")
        parser.add_argument("-r", "--rate", type=float, help="growth rate over forecast period")
        parser.add_argument("-s", "--seed", type=int, help="random seed value")
        parser.add_argument("-c", "--calib", action="store_true", help="whether to use calibrated coeffs")
        parser.add_argument("-o", "--out", action="store_true", help="whether to write output h5")
        parser.add_argument("-p", "--pod", type=str, help="run sim for specific model pod")
        args = parser.parse_args()

        forecast_year = args.year if args.year else 2011
        growth_rate = args.rate if args.rate else .01
        random_seed = args.seed if args.seed else 0
        calibrated = True if args.calib else False
        data_out = True if args.out else False
        pod = args.pod if args.pod else False

        run(forecast_year, growth_rate, random_seed, calibrated, data_out, pod)

    else:
        run()
