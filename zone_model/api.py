import sys
import subprocess
import orca
import models
from flask import Flask, jsonify

app = Flask(__name__)

def run_simulation(forecast_year, seed=None, hh_controls=None, emp_controls=None,
                   scenario_zoning=None, dev_projects=None, subarea_controls=None,
                   skims=None):
    """
    Set up and run a standard simulation.
    """
    if seed:
        np.random.seed(seed)

    models.set_tables(annual_household_control_totals = hh_controls,
                      annual_employment_control_totals = emp_controls,
                      zoning = scenario_zoning,
                      scheduled_development_events = dev_projects,
                      refinement_settings = subarea_controls,
                      tm_skims = skims)

    orca.run(['refresh_h5',
              'build_networks'])
    orca.run(["neighborhood_vars",
              "households_transition_basic",
              "hlcm_simulate"
             ], iter_vars=range(2010, 
                                forecast_year + 1))
    orca.run(['generate_indicators'])

    url_to_results = orca.get_injectable('indicators')
    return url_to_results


@app.route('/model/<int:forecast_year>/<int:seed>/<string:hh_controls>/<string:emp_controls>/<string:scenario_zoning>/<string:dev_projects>/<string:subarea_controls>/<string:skims>', methods=['GET'])
def simulate(forecast_year, seed, hh_controls, emp_controls, scenario_zoning, dev_projects, subarea_controls, skims):
    """
    Model endpoint for launching simulation.
    GET /model/:year/:seed/:hh_controls/:emp_controls/:scenario_zoning/:dev_projects/:subarea_controls

    Parameters
    ----------
    forecast_year : int
        Year to simulate to.
    seed : int
        Random seed.
    hh_controls : str
        Name of household control totals table.
    emp_controls : str
        Name of employment control totals table.
    scenario_zoning : str
        Name of zoning scenario.
    dev_projects : str
        Name of development projects scenario.
    subarea_controls : str
        Name of subarea controls (i.e. refinement settings).
    skims : str
        Name of skims table.
    Returns
    -------
    simulation_result : dict
        Status, and URL to zipfile of simulation results on S3.
    """
    result = run_simulation(forecast_year, hh_controls=hh_controls, emp_controls=emp_controls,
                            scenario_zoning=scenario_zoning, dev_projects=dev_projects,
                            subarea_controls=subarea_controls, skims=skims)
    return jsonify({'status':'Simulation complete', 'url':result})

@app.route('/sim', methods=['GET'])
def sim():
    result = subprocess.check_call([sys.executable, 'simulate.py'])
    return jsonify({'status':'Simulation complete', 'url':result})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
