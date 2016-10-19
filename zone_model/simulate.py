import sys
import time
import argparse
import numpy as np

import orca

import datasources

def run(forecast_year=2011, growth_rate=.01, random_seed=False, data_out=False):
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
    data_out : boolean, optional
        Indicates whether to write output to HDF5.
    Returns
    -------
    _ : None
    """

    # Record start time
    start_time = time.time()
    orca.add_injectable('start_time', start_time)

    # Set value of random seed
    if random_seed:
        np.random.seed(random_seed)

    # Set model parameters
    orca.add_injectable('growth_rate', growth_rate)
    orca.add_injectable('forecast_year',forecast_year)

    # Register models, variables
    import variables
    import models
    
    # Simulate
    if data_out:
        orca.run(datasources.step_sequence(), iter_vars = range(2011, forecast_year + 1), data_out='./data/run_output.h5')
    else:
        orca.run(datasources.step_sequence(), iter_vars = range(2011, forecast_year + 1))

    # Record end time
    end_time = time.time()
    time_elapsed = end_time - start_time
    print 'Simulation duration: %s minutes' % (time_elapsed/60)


if __name__ == '__main__':

    # Run simulation with optional command-line arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-y", "--year", type=int, help="forecast year to simulate to")
        parser.add_argument("-r", "--rate", type=float, help="growth rate over forecast period")
        parser.add_argument("-s", "--seed", type=int, help="random seed value")
        parser.add_argument("-o", "--out", action="store_true", help="whether to write output h5")

        args = parser.parse_args()
        forecast_year = args.year if args.year else 2011
        growth_rate = args.rate if args.rate else .01
        random_seed = args.seed if args.seed else False
        data_out = True if args.out else False

        run(forecast_year, growth_rate, random_seed, data_out)

    else:
        run()
