import orca
import numpy as np
import time
import datasources
import requests
import json

# user vars
base_url = "http://paris.urbansim.com:3000/"
user = 'mbianchetti'
passwd = 'montoto'
mpo = 'MRCOG Testing'


def authenticate(user, passwd, mpo, base_url):
    s = requests.Session()
    url = base_url + "users/login/"
    temp_tk = s.post(url, data={'user': user, 'pass': passwd})
    headers = {'urbancanvastk': temp_tk.text}
    data = {
        'region': json.dumps({
            'id': 1, 'name': mpo
        }),
        'user': user
    }
    url = base_url + "users/regionlogin"
    token = s.post(url, headers=headers, data=data)
    s.headers.update({'urbancanvastk': token.text})
    return s


@orca.step('start_of_year_signal')
def start_of_year_signal(year):
    print 'Year %s starting.' % year


@orca.step('end_of_year_signal')
def end_of_year_signal(year):
    print 'Year %s completed.' % year


@orca.step('sim_progress_by_year')
def sim_progress_by_year(year, forecast_year, start_time):
    now = time.time()
    elapsed = now - start_time
    base_year = 2010
    total_years = forecast_year - base_year
    pct_progress = np.round((year - base_year) / float(total_years), 2)
    time_left = elapsed * (1.0 / pct_progress)
    print("Simulation {0}% complete. Time remaining: {1} seconds.".format(
        int(pct_progress * 100), time_left))


@orca.step('sim_progress_by_step')
def sim_progress_by_step(year, forecast_year, start_time, iter_step, run_id):
    now = time.time()
    elapsed_time = now - start_time
    base_year = 2010
    total_years = forecast_year - base_year
    years_elapsed = year - base_year
    all_steps = datasources.step_sequence()
    model_steps = [x for x in all_steps if x != 'sim_progress_by_step']
    num_steps = len(model_steps)
    total_steps = (num_steps * total_years)
    cur_step_pos = (iter_step.step_num - 1) / 2
    total_step_pos = (years_elapsed - 1.0) * num_steps + cur_step_pos
    pct_progress = np.round((total_step_pos + 1) / total_steps, 2)
    time_left = elapsed_time * (1.0 / pct_progress) - elapsed_time

    s = authenticate(user, passwd, mpo, base_url)
    url = base_url + 'simulation/progress/' + str(run_id) + '/'
    data = {
        'completion': int(pct_progress*100),
        'time_remaining': int(time_left),
        'current_step': int(total_step_pos), #model_steps[cur_step_pos],
        'current_year': int(year)
    }
    progress = s.post(url, data=data)
    if not progress.json()['success']:
        print progress.json()
        raise Exception('Progress tracking failed.')
    else:
        print(data)

    # if total_step_pos != total_steps - 1:
    #     print("TOTAL STEPS: {0}".format(total_steps))
    #     print("CURRENT STEP: {0}".format(model_steps[cur_step_pos]))
    #     print("CURRENT STEP POSITION: {0}".format(total_step_pos))
    #     print("CURRENT YEAR: {0}".format(year))
    #     print("ELAPSED TIME: {0}".format(elapsed_time))
    #     print("SIMULATION {0}% COMPLETE. TIME REMAINING: {1} SECONDS.".format(
    #         int(pct_progress * 100), time_left))
    # else:
    #     print("SIMULATION {0}% COMPLETE.".format(pct_progress * 100))
