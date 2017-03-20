import os
import yaml
import numpy as np
import pandas as pd

import orca
import pandana as pdna
from urbansim.utils import misc
from urbansim.utils import yamlio
from urbansim.utils import networks
from urbansim.models import transition
from urbansim.models import RegressionModel
from urbansim.models import MNLDiscreteChoiceModel
from urbansim.models import SegmentedMNLDiscreteChoiceModel

import utils
import datasources
import variables

from google.cloud import datastore
client = datastore.Client()

geography_base_id = orca.get_injectable('geography_id')

@orca.step('refresh_h5')
def refresh_h5():
    pass

@orca.step('generate_indicators')
def generate_indicators():
    orca.add_injectable('indicators', 'URL_to_results_on_S3')
    pass


@orca.step('build_networks')
def build_networks(store, zones):
    if 'net' not in orca.list_injectables():
        nodes, edges = store.nodes, store.edges
        
        print 'Number of nodes is %s.' % len(nodes)
        print 'Number of edges is %s.' % len(edges)
        net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                           edges[["weight"]])

        precompute_distance = 40000
        print 'Precomputing network for distance %s.' % precompute_distance

        orca.add_injectable("net", net)
        print 'Network precompute starting.'
        net.precompute(precompute_distance)
        print 'Network precompute done.'
        b = zones.to_frame(zones.local_columns)
        b['node_id'] = net.get_node_ids(b['x'], b['y'])
        orca.add_table("zones", b)
        

@orca.step('households_transition_basic')
def households_transition_basic(households):
    growth_rate = orca.get_injectable('growth_rate') if 'growth_rate' in orca.list_injectables() else .01
    print 'Running household transition with %s percent growth rate' % (growth_rate*100.0)
    return utils.simple_transition(households, growth_rate, geography_base_id)
    
@orca.step('jobs_transition_basic')
def jobs_transition_basic(jobs):
    growth_rate = orca.get_injectable('growth_rate') if 'growth_rate' in orca.list_injectables() else .01
    print 'Running employment transition with %s percent growth rate' % (growth_rate*100.0)
    return utils.simple_transition(jobs, growth_rate, geography_base_id)
    
@orca.step('residential_unit_transition_basic')
def residential_unit_transition_basic(residential_units):
    growth_rate = orca.get_injectable('growth_rate') if 'growth_rate' in orca.list_injectables() else .01
    print 'Running dwelling unit transition with %s percent growth rate' % (growth_rate*100.0)
    return utils.simple_transition(residential_units, growth_rate, geography_base_id, set_year_built=True)
    
@orca.step('households_relocation_basic')
def households_relocation_basic(households, settings):
    return utils.simple_relocation(households, .005, geography_base_id)
    
@orca.step('households_relocation_full')
def households_relocation_full(households, household_relocation_rates):
    relocation_rates = household_relocation_rates.to_frame()
    reloc = relocation.RelocationModel(relocation_rates, 'probability_of_relocating')
    hh = households.to_frame(households.local_columns)
    idx_reloc = reloc.find_movers(hh)
    print "Households unplaced by relocation: %d" % len(idx_reloc)
    hh.loc[idx_reloc, geography_base_id] = -1
    orca.add_table('households', hh)

@orca.step('jobs_relocation_basic')
def jobs_relocation_basic(jobs, settings):
    return utils.simple_relocation(jobs, .005, geography_base_id)
    
@orca.step('households_transition_full')
def households_transition_full(households, household_controls, year, persons):
    return utils.full_transition(households, household_controls, year, geography_base_id,
                                 linked_tables={'persons':(persons, 'household_id')})
                                 
@orca.step('jobs_transition_full')
def jobs_transition_full(jobs, employment_controls, year):
    return utils.full_transition(jobs, employment_controls, year, geography_base_id)
    
@orca.step('residential_units_transition_full')
def residential_units_transition_full(residential_units, residential_unit_controls, year):
    return utils.full_transition(residential_units, residential_unit_controls, year,
                                 geography_base_id, set_year_built=True)

@orca.step('start_indicators')
def start_indicators(iter_var, first_year):
    if iter_var == first_year:
        print 'Generating start indicators'
        indicators_df = run_indicators()
        orca.add_table('indicators_base', indicators_df)
        
@orca.step('end_indicators')
def end_indicators(iter_var, last_year):
    if iter_var == last_year:
        print 'Generating end indicators'
        indicators_df = run_indicators()
        orca.add_table('indicators_forecast', indicators_df)
        
@orca.step('scheduled_development_events_model')
def scheduled_development_events_model(zones, jobs, households, residential_units, year):
    ru = residential_units.local
    j = jobs.local
    b = zones.local
    hh = households.local
    
    try:
        redev = orca.get_table('redevelopments').to_frame()
    except:
        print "There are no redevelopment projects in the simulation"
    redev = redev[redev.start_date==year]
    if len(redev.index) > 0:
        print "removing redeveloped units and unplacing agents"
        b.employment_capacity[b.index.isin(redev.block)] = 0
        ru = ru[~ru[geography_base_id].isin(redev.block)]
        j[geography_base_id][j[geography_base_id].isin(redev.block)] = '-1'
        hh[geography_base_id][hh[geography_base_id].isin(redev.block)] = '-1'
    
    rd = orca.get_table('residential_developments').to_frame()
    rd = rd[rd.year_built==year]
    nrd = orca.get_table('nonres_developments').to_frame()
    nrd = nrd[nrd.year_built==year]
    
    rd.index = np.arange(ru.index.max()+1, ru.index.max() + len(rd.index) + 1)
    ru = pd.concat([ru, rd.rename(columns={'block':geography_base_id})])
    orca.add_table('residential_units', ru)
    
    b[geography_base_id] = b.index
    b = pd.merge(b, nrd, how='left', left_index=True, right_on='block', suffixes=('','_x'))
    b.employment_capacity[b.employment_capacity_x.notnull()] = b.employment_capacity[b.employment_capacity_x.notnull()] + b.employment_capacity_x[b.employment_capacity_x.notnull()]
    b.employment_capacity = b.employment_capacity.astype('int')
    b.set_index(geography_base_id, inplace=True)
    b = b.drop(['block','year_built','employment_capacity_x'], axis=1)
    orca.add_table('zones', b)

@orca.step('refinement_model')
def refinement_model(refinements, year):
    refinements = refinements.to_frame()
    refinements = refinements[refinements.year==year]
    if len(refinements.index)==0:
        print "no refinement targets for %s" % year
        return 1
    
    # calculate deficits and surpluses for each set of refinement zones
    refinements['difference'] = 0
    diffs = []
    for i, row in refinements.iterrows():
        df = orca.get_table(row.ref_type).local
        df = df.query(row.filters) if len(row.filters)>0 else df
        df = df[df[geography_base_id].isin(row.block_ids)]
        diff = row[row.ref_type] - df[geography_base_id].count()
        diffs.append(diff)
        if diff < 0:
            print 'surplus of %d %s in zones %s'  %(np.abs(diff), row.ref_type, row.block_ids)
        elif diff > 0:
            print 'deficit of %d %s in zones %s' %(np.abs(diff), row.ref_type, row.block_ids)
        else:
            print 'zones %s are right on target!' %(row.block_ids)
    refinements.difference = diffs

    # unplace surplus residential_units, households, jobs and place in other zones
    to_unplace = refinements[refinements.difference < 0]
    for i, row in to_unplace.iterrows():
        utils.unplace(row.ref_type, np.abs(row.difference), row.block_ids, row.filters)
        if row.ref_type == 'residential_units':
            utils.place_res_units(list(orca.get_table('zones').index.difference(row.block_ids)))
        elif row.ref_type == 'households':
            utils.place_households(list(orca.get_table('zones').index.difference(row.block_ids)))
        else:
            utils.place_jobs(list(orca.get_table('zones').index.difference(row.block_ids)))
    
    # unplace residential_units from outside deficit zones and relocate to those zones
    du_deficit = refinements[(refinements.difference > 0) & (refinements.ref_type=='residential_units')]
    zones = []
    for i, row in du_deficit.iterrows():
        zones.extend(row.block_ids)
    all_zones = orca.get_table('zones').index - zones
    for i, row in du_deficit.iterrows():
        utils.unplace(row.ref_type, row.difference, all_zones, row.filters)
        utils.place_res_units(row.block_ids)

    # place households that were unplaced along with res units in the previous step
    utils.place_households(zones)

    # unplace households from outside deficit areas and place them into target areas
    hh_deficit = refinements[(refinements.difference > 0) & (refinements.ref_type=='households')]
    zones = []
    for i, row in hh_deficit.iterrows():
        zones.extend(row.block_ids)
    all_zones = orca.get_table('zones').index - zones
    for i, row in hh_deficit.iterrows():
        utils.unplace(row.ref_type, row.difference, all_zones, row.filters)
        blk = orca.get_table('zones').to_frame(['vacant_residential_units','residential_units'])
        if blk.vacant_residential_units[blk.index.isin(row.block_ids)].sum() < row.difference:
            print 'not enough vacant residential units to place all households'
        utils.place_households(row.block_ids)

    # unplace jobs from outside deficit areas and place them into target areas
    job_deficit = refinements[(refinements.difference > 0) & (refinements.ref_type=='jobs')]
    zones = []
    for i, row in job_deficit.iterrows():
        zones.extend(row.block_ids)
    all_zones = orca.get_table('zones').index - zones
    for i, row in job_deficit.iterrows():
        utils.unplace(row.ref_type, row.difference, all_zones, row.filters)
        blk = orca.get_table('zones').to_frame(['vacant_job_spaces','job_spaces'])
        if blk.vacant_job_spaces[blk.index.isin(row.block_ids)].sum() < row.difference:
            print 'not enough vacant job spaces to place all jobs'
        utils.place_jobs(row.block_ids)                               
                                          

@orca.step('calibration_indicators')
def calib_indicators(store, zones):
    year = orca.get_injectable('year')

    orca.clear_cache()
    calib_geography_id = orca.get_injectable('model_structure')['calibration']['calibration_geography_id']

    # Simulated agents
    households = orca.get_table('households').to_frame(columns = calib_geography_id)
    jobs = orca.get_table('jobs').to_frame(columns = calib_geography_id)
    du = orca.get_table('residential_units').to_frame(columns = calib_geography_id)

    # Simulated agents by calibration geography
    hh_by_geog = households.groupby(calib_geography_id).size()
    emp_by_geog = jobs.groupby(calib_geography_id).size()
    du_by_geog = du.groupby(calib_geography_id).size()

    # Base agents
    households_base = store['households']
    jobs_base = store['jobs']
    du_base = store['residential_units']
    households_base[calib_geography_id] = misc.reindex(zones[calib_geography_id], households_base[geography_base_id]).fillna(0)
    jobs_base[calib_geography_id] = misc.reindex(zones[calib_geography_id], jobs_base[geography_base_id]).fillna(0)
    du_base[calib_geography_id] = misc.reindex(zones[calib_geography_id], du_base[geography_base_id]).fillna(0)

    # Base agents by calibration geography
    hh_by_geog_base = households_base.groupby(calib_geography_id).size()
    emp_by_geog_base = jobs_base.groupby(calib_geography_id).size()
    du_by_geog_base = du_base.groupby(calib_geography_id).size()

    # Change in agents by calibration geography
    hh_diff = hh_by_geog - hh_by_geog_base
    emp_diff = emp_by_geog - emp_by_geog_base
    du_diff = du_by_geog - du_by_geog_base

    indicators_df = pd.DataFrame({'hlcm':hh_diff, 'elcm':emp_diff, 'rdplcm':du_diff})
    indicators_df = indicators_df*1.0/indicators_df.sum()
    print indicators_df
    indicators_df.to_csv('./data/calib_indicators.csv')
    return indicators_df   


def read_yaml_config(file_name):
    with open(os.path.join(misc.configs_dir(), file_name)) as f:
        config = yaml.load(f)
        return config
    

def write_yaml_config(file_name, data):
    with open(os.path.join(misc.configs_dir(), file_name), 'w') as outfile:
        outfile.write(yaml.dump(data, default_flow_style=True) )


def make_lcm_func(model_name, yaml_file, agents_name, alts_name, alts_id_name, supply_attrib, remaining_capacity_attrib):
    @orca.step(model_name)
    def func():
        agents = orca.get_table(agents_name)
        alts = orca.get_table(alts_name)
        print yaml_file
        return utils.lcm_simulate(yaml_file, agents, alts,
                                  [],
                                  alts_id_name, supply_attrib,
                                  remaining_capacity_attrib)
    return func


def make_repm_func(model_name, yaml_file, dep_var, observations_name):
    """
    Generator function for block REPMs.
    """

    @orca.step(model_name)
    def func():
        obs = orca.get_table(observations_name)
        print yaml_file
        return utils.hedonic_simulate(yaml_file, obs,
                                      [], dep_var)
    return func


def get_selected_models_by_version(model_version_id):
    query = client.query(kind='Spec')
    query.add_filter('run_id', '=', model_version_id)
    query.add_filter('selected', '=', True)
    return list(query.fetch())


models_version_id = '70320150328dnh'  ## pull this into config file
models = get_selected_models_by_version(models_version_id)

for model in models:
    model_type = model['model_type']
    model_name = model['model_name']
    yaml_cfg = model['yaml']
    dict_cfg = yaml.load(yaml_cfg)
    
    if model_type == 'location':
        agents = model['agents']
        alternatives = model['alternatives']
        alternatives_id_name = dict_cfg['choice_column']
        supply_variable = agents + '_spaces'
        vacant_variable = 'vacant_' + supply_variable
        make_lcm_func(model_name,  yaml_cfg, agents, alternatives, alternatives_id_name, 
                      supply_variable, vacant_variable)

    elif model_type == 'regression':
        dep_var = model['dep_var']
        obs_name = model['observations_name']
        make_repm_func(model_name, yaml_cfg, dep_var, obs_name)
        
    else:
        print 'Model type "%s" unrecognized' % model_type

