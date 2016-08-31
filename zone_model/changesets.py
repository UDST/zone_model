import os
import sys
import time
import pickle
import subprocess

import numpy as np
import pandas as pd

import orca

def run_subprocess(filename):
    """"Run Python file relative to script without blocking."""
    python = sys.executable
    root_path = os.path.dirname(__file__)
    path = os.path.join(root_path, filename)
    return subprocess.Popen([python, path])

@orca.step('export_change_sets')
def export_change_sets(year, change_sets, track_changes,
                       households, jobs, residential_units):

    if track_changes:
        change_sets = change_sets[year]

        added = change_sets['added']
        updated = change_sets['updated']

        change_sets_for_export = {}
        change_sets_for_export['added'] = {}
        change_sets_for_export['updated'] = {}

        def swap_records_for_idx(tbl, idx):
            dfw = orca.get_table(tbl)
            new = dfw.to_frame(dfw.local_columns).loc[idx]
            return new

        for agent in ['households', 'jobs', 'residential_units']:
            new_agents_idx = [added[add_set] for add_set in added.keys() if add_set[0] == agent]
            
            new_agents = [swap_records_for_idx(agent, idx) for idx in new_agents_idx if len(idx) > 0]
            if len(new_agents) == 1:
                new_agents = new_agents[0]
            else:
                pass
                #implement concat if multiple models are producing new agents

            change_sets_for_export['added'][agent] = new_agents

        updates_to_export = [update_set for update_set in updated.keys() if update_set[0] == 'zones']
        for update in updates_to_export:
            change_sets_for_export['updated'][('repm', update[0], update[1])] = updated[update]

        # Serialize change set
        try:
            pod = orca.get_injectable('pod')
            pkl_path = ('./data/change_sets_%s_' % year) + pod + '.pkl'
        except:
            pkl_path = './data/change_sets_%s.pkl' % year

        pickle.dump(change_sets_for_export, open(pkl_path, "wb"))
        # Spawn subprocess to export change set to app server
        # run_subprocess('export_change_sets.py')

from urbansim.developer import developer
def merge_dfs(a, b):
    dev = developer.Developer([])
    merged = dev.merge(a, b)
    merged.index.name = a.index.name
    return merged

@orca.step('incorporate_change_sets')
def incorporate_change_sets(year, pod, pods):
    print 'Incorporating change sets from other pods'
    current_pod = pod
    if current_pod != 'network':
        print current_pod
        print pods
        pods = list(pods)
        pods.remove(current_pod)
    additions = {}
    updates = {}
    for pod in pods:
        change_sets = pickle.load(open("./data/change_sets_%s_%s.pkl" % (year, pod)))

        for add_batch in change_sets['added'].items():
            agent_type = add_batch[0]
            agent_data = add_batch[1]
            if len(agent_data) > 0:
                additions[(pod, agent_type)] = agent_data

        for update_batch in change_sets['updated'].items():
            if update_batch[0][0] == 'repm':
                updated_table = update_batch[0][1]
                updated_column = update_batch[0][2]
                updated_data = update_batch[1]
                if len(updated_data) > 0:
                    updates[(pod, updated_table, updated_column)] = updated_data

    for pod_additions in additions.items():
        pod = pod_additions[0][0]
        agent_type = pod_additions[0][1]
        agent_data = pod_additions[1]
        if pod != current_pod:
            ## Merge in changes if pod of change set not equals current process's pod
            print 'Incorporating %s change set into the %s process.' % (agent_type, current_pod)
            old_dfw = orca.get_table(agent_type)
            old_df = old_dfw.to_frame(old_dfw.local_columns)
            print '    Previous number of agents: %s' % len(old_df)
            merged = merge_dfs(old_df, agent_data)
            print '    Updated number of agents: %s' % len(merged)
            print merged.head()
            orca.add_table(agent_type, merged)

    for pod_update in updates.items():
        pod = pod_update[0][0]
        updated_table = pod_update[0][1]
        updated_column = pod_update[0][2]
        updated_data = pod_update[1]
        if pod != current_pod:
            print 'Incorporating %s update from %s into the %s process' % (updated_column, pod, current_pod)
            orca.add_column(updated_table, updated_column, updated_data)

@orca.step('redis_checkin1')
def redis_checkin1(redis_conn, number_of_pods):
    redis_conn.incr('pod_count') # Number of pods that have finished.  Increment when each pod finishes, reset each year.
    # todo:  add pod-specific checkins for pre-calculation of next year's variables

    ## Wait until all pods finish exporting change sets before proceeding with incorporation of change sets
    while True:
        time.sleep(.1)
        pod_count = redis_conn.get('pod_count')
        if pod_count == str(number_of_pods):
            break

@orca.step('redis_checkin2')
def redis_checkin2(redis_conn, year):
    while True:
        time.sleep(.1)
        sim_year = redis_conn.get('year')
        if int(sim_year) == (year + 1):
            break
    print 'Year %s done!' % year 
