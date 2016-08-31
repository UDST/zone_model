import time
import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc
from variable_generators import generators

import datasources
geography_base_id = orca.get_injectable('geography_id')

#####################
# ZONE VARIABLES 1
#####################

@orca.column('zones', 'all_zones', cache=True)
def all_zones(zones):
    return pd.Series(np.ones(len(zones.residential_units)).astype('int32'), index = zones.index)

@orca.column('zones', 'residential_units', cache=False)
def residential_units(zones, residential_units):
    du_by_zone = residential_units[geography_base_id].groupby(residential_units[geography_base_id]).size()
    return pd.Series(index=zones.index, data=du_by_zone).fillna(0)
    
@orca.column('zones', 'households', cache=False)
def zones_households(zones, households):
    hh_by_zone = households[geography_base_id].groupby(households[geography_base_id]).size()
    return pd.Series(index=zones.index, data=hh_by_zone).fillna(0)
    
@orca.column('zones', 'jobs', cache=False)
def zones_jobs(zones, jobs):
    jobs_by_zone = jobs[geography_base_id].groupby(jobs[geography_base_id]).size()
    return pd.Series(index=zones.index, data=jobs_by_zone).fillna(0)

@orca.column('zones', 'ln_residential_units', cache=True, cache_scope='iteration')
def ln_residential(zones):
    return zones.residential_units.apply(np.log1p)
    
@orca.column('zones', 'z_id', cache=True)
def z_id(zones):
    return zones.index

@orca.column('zones', 'acres', cache=True)
def acres(zones):
    return zones.square_meters_land/4046.86


#####################
# HOUSEHOLD VARIABLES
#####################

@orca.column('households', 'income_quartile', cache=True)
def income_quartile(households):
    s = pd.Series(pd.qcut(households.income, 4, labels=False),
                  index=households.index)
    # e.g. convert income quartile from 0-3 to 1-4
    s = s.add(1)
    return s

@orca.column('households', 'age_cat', cache=True)
def age_cat(households):
    return 1*(households.age_of_head<=35) + 2*(households.age_of_head>35)*(households.age_of_head<=60) + 3*(households.age_of_head>60)

@orca.column('households', 'hhsize3plus', cache=True)
def hhsize3plus(households):
    return (households.persons>2).astype('int32')

@orca.column('households', 'hhsize2', cache=True)
def hhsize2(households):
    return (households.persons==2).astype('int32')

@orca.column('households', 'young', cache=True)
def young(households):
    return (households.age_cat==1).astype('int32')

@orca.column('households', 'middle_age', cache=True)
def middle_age(households):
    return (households.age_cat==2).astype('int32')

@orca.column('households', 'old', cache=True)
def old(households):
    return (households.age_cat==3).astype('int32')

@orca.column('households', 'with_child', cache=True)
def with_child(households):
    return (households.children>0).astype('int32')

@orca.column('households', 'with_car', cache=True)
def with_car(households):
    return (households.cars>0).astype('int32')
    
@orca.column('households', 'node_id', cache=True, cache_scope='iteration')
def node_id(households, zones):
    return misc.reindex(zones.node_id, households[geography_base_id])
    
@orca.column('households', 'tract_id', cache=True, cache_scope='iteration')
def tract_id(households, zones):
    return misc.reindex(zones.tract_id, households[geography_base_id])
    
@orca.column('households', 'block_group_id', cache=True, cache_scope='iteration')
def block_group_id(households, zones):
    return misc.reindex(zones.block_group_id, households[geography_base_id])
    
@orca.column('households', 'puma10_id', cache=True, cache_scope='iteration')
def puma10_id(households, zones):
    return misc.reindex(zones.puma10_id, households[geography_base_id])

@orca.column('households', 'county_id', cache=True, cache_scope='iteration')
def county_id(households, zones):
    return misc.reindex(zones.county_id, households[geography_base_id]).fillna(0)
    
#####################
# JOB VARIABLES
#####################

@orca.column('jobs', 'aggr_sector_id', cache=True)
def aggr_sector_id(jobs):
    jobs = jobs.to_frame(columns=['sector_id'])
    jobs['aggr_sector_id'] = 0 # Management, Public Administration
    jobs.aggr_sector_id[np.in1d(jobs.sector_id, [11, 21, 22, 23, 3133])] = 1 # basic industries (agr, forestry/fishing, mining, construction, manufacturing)
    jobs.aggr_sector_id[np.in1d(jobs.sector_id, [42, 4849])] = 2 # Transportation, Communications & Public Utilities, Warehousing
    jobs.aggr_sector_id[np.in1d(jobs.sector_id, [4445, 72])] = 3 # Retail Trade
    jobs.aggr_sector_id[np.in1d(jobs.sector_id, [52, 53])] = 4 # Finance, Insurance, and Real Estate
    jobs.aggr_sector_id[np.in1d(jobs.sector_id, [51, 54, 56, 61, 62, 71, 81])] = 5 # Services
    return jobs.aggr_sector_id

@orca.column('jobs', 'node_id', cache=True, cache_scope='iteration')
def node_id(jobs, zones):
    return misc.reindex(zones.node_id, jobs[geography_base_id]).fillna(0)

@orca.column('jobs', 'block_group_id', cache=True, cache_scope='iteration')
def block_group_id(jobs, zones):
    return misc.reindex(zones.block_group_id, jobs[geography_base_id]).fillna(0)
    
@orca.column('jobs', 'tract_id', cache=True, cache_scope='iteration')
def tract_id(jobs, zones):
    return misc.reindex(zones.tract_id, jobs[geography_base_id]).fillna(0)
    
@orca.column('jobs', 'puma10_id', cache=True, cache_scope='iteration')
def puma10_id(jobs, zones):
    return misc.reindex(zones.puma10_id, jobs[geography_base_id]).fillna(0)

@orca.column('jobs', 'county_id', cache=True, cache_scope='iteration')
def county_id(jobs, zones):
    return misc.reindex(zones.county_id, jobs[geography_base_id]).fillna(0)


#####################
# ZONE VARIABLES 2
#####################

@orca.injectable(geography_base_id, cache=True)
def geog_base_ids(zones):
    zones = zones.to_frame(zones.local_columns)
    return zones.index

@orca.injectable('zone_ids', cache=True)
def zone_ids(zones):
    zones = zones.to_frame(zones.local_columns)
    return zones.index

@orca.column('zones', 'job_spaces', cache=True)
def job_spaces(zones, jobs):
    return zones.employment_capacity  # Zoning placeholder:  job capacity

@orca.column('zones', 'vacant_job_spaces', cache=False) # The ELCM capacity variable
def vacant_job_spaces(zones, jobs):
    return zones.job_spaces.sub(
        jobs[geography_base_id].value_counts(), fill_value=0)
        
@orca.column('zones', 'du_spaces', cache=True)
def du_spaces(zones):
    return zones.residential_unit_capacity # The RDPLCM capacity variable

@orca.column('zones', 'mode_res_building_type', cache=True, cache_scope='iteration')
def mode_res_building_type(zones, residential_units):
    mode_btype = residential_units.building_type_id.groupby(residential_units[geography_base_id]).agg(lambda x:x.value_counts().index[0])
    mode_btype = pd.Series(index=zones.index, data=mode_btype).fillna(0)
    return mode_btype

@orca.column('zones', 'res_btype_mode1', cache=True, cache_scope='iteration')
def res_btype_mode1(zones):
    return (zones.mode_res_building_type == 1).astype('int32')

@orca.column('zones', 'res_btype_mode2', cache=True, cache_scope='iteration')
def res_btype_mode2(zones):
    return (zones.mode_res_building_type == 2).astype('int32')

@orca.column('zones', 'res_btype_mode3', cache=True, cache_scope='iteration')
def res_btype_mode3(zones):
    return (zones.mode_res_building_type == 3).astype('int32')

@orca.column('zones', 'res_btype_mode4', cache=True, cache_scope='iteration')
def res_btype_mode4(zones):
    return (zones.mode_res_building_type == 4).astype('int32')

@orca.column('zones', 'own_singlfam_post2010', cache=True, cache_scope='iteration')
def own_singlfam_post2010(zones, residential_units):
    residential_units = residential_units.to_frame(residential_units.local_columns)
    resunits = residential_units[(residential_units.building_type_id == 1) & (residential_units.year_built > 2010)].groupby(geography_base_id).size()
    resunits = pd.Series(index=zones.index, data=resunits).fillna(0)
    return resunits

@orca.column('zones', 'own_multifam_post2010', cache=True, cache_scope='iteration')
def own_multifam_post2010(zones, residential_units):
    residential_units = residential_units.to_frame(residential_units.local_columns)
    resunits = residential_units[(residential_units.building_type_id == 2) & (residential_units.year_built > 2010)].groupby(geography_base_id).size()
    resunits = pd.Series(index=zones.index, data=resunits).fillna(0)
    return resunits

@orca.column('zones', 'rnt_singlfam_post2010', cache=True, cache_scope='iteration')
def rnt_singlfam_post2010(zones, residential_units):
    residential_units = residential_units.to_frame(residential_units.local_columns)
    resunits = residential_units[(residential_units.building_type_id == 3) & (residential_units.year_built > 2010)].groupby(geography_base_id).size()
    resunits = pd.Series(index=zones.index, data=resunits).fillna(0)
    return resunits

@orca.column('zones', 'rnt_multifam_post2010', cache=True, cache_scope='iteration')
def rnt_multifam_post2010(zones, residential_units):
    residential_units = residential_units.to_frame(residential_units.local_columns)
    resunits = residential_units[(residential_units.building_type_id == 4) & (residential_units.year_built > 2010)].groupby(geography_base_id).size()
    resunits = pd.Series(index=zones.index, data=resunits).fillna(0)
    return resunits

@orca.column('zones', 'vacant_du_spaces', cache=False)
def vacant_du_spaces(zones, residential_units):
    return zones.du_spaces.sub(
        residential_units[geography_base_id].value_counts(), fill_value=0)
        
@orca.column('zones', 'vacant_residential_units', cache=False)  ##The HLCM capacity variable
def vacant_residential_units(zones, households):
    return zones.residential_units.sub(
        households[geography_base_id].value_counts(), fill_value=0)


#####################
#     PUMA
#####################
@orca.column('pumas', 'residential_units', cache=True, cache_scope='iteration')
def residential_units(zones):
    return zones.residential_units.groupby(zones.puma10_id).sum()

@orca.column('pumas', 'households', cache=True, cache_scope='iteration')
def households(households):
    return households.serialno.groupby(households.puma10_id).size()
    
@orca.column('pumas', 'vacant_residential_units', cache=True, cache_scope='iteration')
def vacant_residential_units(pumas):
    return pumas.residential_units - pumas.households

@orca.column('pumas', 'residential_vacancy', cache=True, cache_scope='iteration')
def residential_vacancy(pumas):
    return pumas.households*1.0/pumas.residential_units

@orca.column('pumas', 'jobs', cache=True, cache_scope='iteration')
def jobs(jobs):
    return jobs.sector_id.groupby(jobs.puma10_id).size()

@orca.column('pumas', 'job_household_ratio', cache=True, cache_scope='iteration')
def job_household_ratio(pumas):
    return pumas.jobs*1.0/pumas.households
    
    
#######################
#     RESIDENTIAL UNITS
#######################
    
@orca.column('residential_units', 'rent', cache=True, cache_scope='iteration')
def rent(residential_units, zones):

    zones = zones.to_frame(columns=['res_rents', 'tract_id', 'puma10_id'])

    tract_vals = pd.DataFrame(index=np.unique(zones.tract_id), data=zones[zones.res_rents > 0].groupby('tract_id').res_rents.median()).fillna(0)
    puma_vals = pd.DataFrame(index=np.unique(zones.puma10_id), data=zones[zones.res_rents > 0].groupby('puma10_id').res_rents.median()).fillna(0)

    zones['tract_vals'] = misc.reindex(tract_vals.res_rents, zones.tract_id)
    zones['puma_vals'] = misc.reindex(puma_vals.res_rents, zones.puma10_id)

    zones.res_rents[zones.res_rents == 0] = zones.tract_vals[zones.res_rents == 0]
    zones.res_rents[zones.res_rents == 0] = zones.puma_vals[zones.res_rents == 0]

    return misc.reindex(zones.res_rents, residential_units[geography_base_id])
    
@orca.column('residential_units', 'value', cache=True, cache_scope='iteration')
def value(residential_units, zones):

    zones = zones.to_frame(columns=['res_values', 'tract_id', 'puma10_id'])

    tract_vals = pd.DataFrame(index=np.unique(zones.tract_id), data=zones[zones.res_values > 0].groupby('tract_id').res_values.median()).fillna(0)
    puma_vals = pd.DataFrame(index=np.unique(zones.puma10_id), data=zones[zones.res_values > 0].groupby('puma10_id').res_values.median()).fillna(0)

    zones['tract_vals'] = misc.reindex(tract_vals.res_values, zones.tract_id)
    zones['puma_vals'] = misc.reindex(puma_vals.res_values, zones.puma10_id)

    zones.res_values[zones.res_values == 0] = zones.tract_vals[zones.res_values == 0]
    zones.res_values[zones.res_values == 0] = zones.puma_vals[zones.res_values == 0]

    return misc.reindex(zones.res_values, residential_units[geography_base_id])

@orca.column('residential_units', 'node_id', cache=True, cache_scope='iteration')
def node_id(residential_units, zones):
    return misc.reindex(zones.node_id, residential_units[geography_base_id]).fillna(0)

@orca.column('residential_units', 'block_group_id', cache=True, cache_scope='iteration')
def block_group_id(residential_units, zones):
    return misc.reindex(zones.block_group_id, residential_units[geography_base_id]).fillna(0)

@orca.column('residential_units', 'tract_id', cache=True, cache_scope='iteration')
def tract_id(residential_units, zones):
    return misc.reindex(zones.tract_id, residential_units[geography_base_id]).fillna(0)

@orca.column('residential_units', 'puma10_id', cache=True, cache_scope='iteration')
def puma10_id(residential_units, zones):
    return misc.reindex(zones.puma10_id, residential_units[geography_base_id]).fillna(0)

@orca.column('residential_units', 'county_id', cache=True, cache_scope='iteration')
def county_id(residential_units, zones):
    return misc.reindex(zones.county_id, residential_units[geography_base_id]).fillna(0)


#####################
# NODE VARIABLES
#####################
for access_var in orca.get_injectable('neighborhood_vars')['variable_definitions']:
    # print access_var
    generators.make_access_var(access_var['name'], 
                               access_var['dataframe'],
                               radius = access_var['radius'],
                               decay = access_var.get('decay', 'linear'),
                               agg_function = access_var.get('aggregation', 'sum'),
                               log = True if 'apply' in access_var.keys() else False,
                               target_variable = access_var.get('varname', False),
                               filters = access_var.get('filters', False))

#######################
#     AUTOGENERATED
#######################

def make_agg_var(agent, geog, geog_id, var_to_aggregate, agg_function):
    """
    Generator function for aggregation variables. Registers with orca.
    """
    var_name = agg_function + '_' + var_to_aggregate
    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        agents = orca.get_table(agent)
        print 'Calculating %s of %s for %s' % (var_name, agent, geog)

        groupby = agents[var_to_aggregate].groupby(agents[geog_id])
        if agg_function == 'mean':
            values = groupby.mean().fillna(0)
        if agg_function == 'median':
            values = groupby.median().fillna(0)
        if agg_function == 'std':
            values = groupby.std().fillna(0)
        if agg_function == 'sum':
            values = groupby.sum().fillna(0)

        locations_index = orca.get_table(geog).index
        series = pd.Series(data=values, index=locations_index)

        # Fillna.  For certain functions, must add other options, like puma value or neighboring value
        if agg_function == 'sum':
            series = series.fillna(0)
        else:
            series = series.fillna(method='ffill')
            series = series.fillna(method='bfill')

        return series

    return func

def make_disagg_var(from_geog_name, to_geog_name, var_to_disaggregate, from_geog_id_name):
    """
    Generator function for disaggregating variables. Registers with orca.
    """
    var_name = from_geog_name + '_' + var_to_disaggregate
    @orca.column(to_geog_name, var_name, cache=True, cache_scope='iteration')
    def func():
        print 'Disaggregating %s to %s from %s' % (var_to_disaggregate, to_geog_name, from_geog_name)

        from_geog = orca.get_table(from_geog_name)
        to_geog = orca.get_table(to_geog_name)
        return misc.reindex(from_geog[var_to_disaggregate], to_geog[from_geog_id_name]).fillna(0)

    return func

def make_disagg_var(from_geog_name, to_geog_name, var_to_disaggregate, from_geog_id_name, name_based_on_geography=True):
    """
    Generator function for disaggregating variables. Registers with orca.
    """
    if name_based_on_geography:
        var_name = from_geog_name + '_' + var_to_disaggregate
    else:
        var_name = var_to_disaggregate
    @orca.column(to_geog_name, var_name, cache=True, cache_scope='iteration')
    def func():
        print 'Disaggregating %s to %s from %s' % (var_to_disaggregate, to_geog_name, from_geog_name)

        from_geog = orca.get_table(from_geog_name)
        to_geog = orca.get_table(to_geog_name)
        return misc.reindex(from_geog[var_to_disaggregate], to_geog[from_geog_id_name]).fillna(0)

    return func

def node_disagg_var(from_geog_name, to_geog_name, var_to_disaggregate, from_geog_id_name, name_based_on_geography=True):
    """
    Generator function for distributed accessibility variables. Registers with orca.
    """
    r = orca.get_injectable('redis_conn')
    if name_based_on_geography:
        var_name = from_geog_name + '_' + var_to_disaggregate
    else:
        var_name = var_to_disaggregate
    @orca.column(to_geog_name, var_name, cache=True, cache_scope='iteration')
    def func():
        print 'Loading %s from distributed cache.  Waiting for results to be ready' % var_name
        while True:
            time.sleep(.1)
            result = r.get(var_name)
            if (result != 'Not calculated') & (result is not None):
                print 'Ready!'
                series = pd.read_msgpack(result)
                break
        return series
    return func

def make_size_var(agent, geog, geog_id):
    """
    Generator function for size variables. Registers with orca.
    """
    var_name = 'total_' + agent
    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        agents = orca.get_table(agent)
        print 'Calculating number of %s for %s' % (agent, geog)

        size = agents[geog_id].value_counts()

        locations_index = orca.get_table(geog).index
        series = pd.Series(data=size, index=locations_index)
        series = series.fillna(0)

        return series

    return func

def make_proportion_var(agent, geog, geog_id, target_variable, target_value):
    """
    Generator function for proportion variables. Registers with orca.
    """
    var_name = 'prop_%s_%s'%(target_variable, int(target_value))
    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        agents = orca.get_table(agent).to_frame(columns=[target_variable, geog_id])
        locations = orca.get_table(geog)
        print 'Calculating proportion %s %s for %s' % (target_variable, target_value, geog)

        agent_subset = agents[agents[target_variable] == target_value]
        series = agent_subset.groupby(geog_id).size()*1.0/locations['total_' + agent]
        series = series.fillna(0)
        return series

    return func

def make_dummy_variable(geog_var, geog_id):
    """
    Generator function for spatial dummy. Registers with orca.
    """
    try:
        var_name = geog_var + '_is_' + str(geog_id)
    except:
        var_name = geog_var + '_is_' + str(int(geog_id))
    @orca.column('zones', var_name, cache=True, cache_scope='iteration')
    def func():
        zones = orca.get_table('zones')
        return (zones[geog_var] == geog_id).astype('int32')

    return func

def make_ratio_var(agent1, agent2, geog):
    """
    Generator function for ratio variables. Registers with orca.
    """
    var_name = 'ratio_%s_to_%s'%(agent1, agent2)
    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        locations = orca.get_table(geog)
        print 'Calculating ratio of %s to %s for %s' % (agent1, agent2, geog)

        series = locations['total_' + agent1]*1.0/(locations['total_' + agent2] + 1.0)
        series = series.fillna(0)
        return series

    return func

def make_density_var(agent, geog):
    """
    Generator function for density variables. Registers with orca.
    """
    var_name = 'density_%s'%agent
    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        locations = orca.get_table(geog)
        print 'Calculating density of %s for %s' % (agent, geog)
        if geog != 'zones':
            series = locations['total_' + agent]*1.0/(locations['sum_acres'] + 1.0)
        else:
            series = locations['total_' + agent]*1.0/(locations['acres'] + 1.0)
        series = series.fillna(0)
        return series

    return func


aggregation_functions = ['mean', 'median', 'std', 'sum']

geographic_levels = [('block_groups', 'block_group_id'),
                     ('tracts', 'tract_id'),
                     ('pumas', 'puma10_id'),
                     ('counties', 'county_id'),
                     ('zones', geography_base_id)]

variables_to_aggregate = {'households':['persons', 'cars', 'income', 'race_of_head', 'age_of_head',
                                        'workers', 'children', 'tenure', 'recent_mover'],
                          'jobs':['sector_id'],
                          'residential_units':['year_built', 'building_type_id', 'rent', 'value'],
                          'zones':['x','y','own_multifam_post2010','own_singlfam_post2010',
                                    'rnt_multifam_post2010','rnt_singlfam_post2010', 'acres']
                          }
discrete_variables = {'households':['persons', 'cars', 'race_of_head', 'workers', 'children', 'tenure', 'recent_mover'],
                          'jobs':['sector_id', 'aggr_sector_id'],
                          'residential_units':['year_built', 'building_type_id'],
                          }
sum_vars = ['persons', 'cars', 'workers', 'children', 'recent_mover', 'acres']

geog_vars_to_dummify = ['puma10_id', 'county_id']

# Aggregation vars
generated_variables = set([])

for agent in variables_to_aggregate.keys():
    # print agent
    for geography in geographic_levels:
        # print '  ' + geography[0]
        geography_name = geography[0]
        geography_id = geography[1]
        if geography_name != agent:

            #Define size variables
            make_size_var(agent, geography_name, geography_id)
            generated_variables.add('total_' + agent)

            #Define attribute variables
            variables = variables_to_aggregate[agent]
            for var in variables:
                # print '    ' + var
                for aggregation_function in aggregation_functions:
                    # print '      ' + aggregation_function

                    if aggregation_function == 'sum':
                        if var in sum_vars:
                            make_agg_var(agent, geography_name, geography_id, var, aggregation_function)
                            generated_variables.add(aggregation_function + '_' + var)

                    else:
                        make_agg_var(agent, geography_name, geography_id, var, aggregation_function)
                        generated_variables.add(aggregation_function + '_' + var)

# Define prop_X_X variables
for agent in discrete_variables.keys():
    agents = orca.get_table(agent)
    discrete_vars = discrete_variables[agent]
    for var in discrete_vars:
        agents_by_cat = agents[var].value_counts()
        cats_to_measure = agents_by_cat[agents_by_cat > 5000].index.values
        for cat in cats_to_measure:
            for geography in geographic_levels:
                geography_name = geography[0]
                geography_id = geography[1]
                make_proportion_var(agent, geography_name, geography_id, var, cat)
                generated_variables.add('prop_%s_%s'%(var, int(cat)))

# Define ratio variables
for geography in geographic_levels:
    # print geography
    geography_name = geography[0]

    # Jobs-housing balance
    make_ratio_var('jobs', 'households', geography_name)
    generated_variables.add('ratio_jobs_to_households')

    # Residential occupancy rate
    make_ratio_var('households', 'residential_units', geography_name)
    generated_variables.add('ratio_households_to_residential_units')

    # Density
    for agent in discrete_variables.keys():
        make_density_var(agent, geography_name)
        generated_variables.add('density_%s'%agent)

# Define disaggregation vars  ######Add nodal variable disaggregation!!
for geography in geographic_levels:
    # print geography
    geography_name = geography[0]
    geography_id = geography[1]
    if geography_name != 'zones':
        for var in generated_variables:
            # print '  ' + var
            make_disagg_var(geography_name, 'zones', var, geography_id)

# Disaggregate node vars
for var in orca.get_table('nodes').columns:
    if var not in ['x', 'y']:
        if orca.get_injectable('multiprocess'):
            if not orca.get_injectable('process_has_pandana'):
                node_disagg_var('nodes', 'zones', var, 'node_id')
            else:
                make_disagg_var('nodes', 'zones', var, 'node_id')
        else:
            make_disagg_var('nodes', 'zones', var, 'node_id')

# Define geographic dummies
for geog_var in geog_vars_to_dummify:
    # print geog_var
    geog_ids = np.unique(orca.get_table('zones')[geog_var]) # works for counties too
    for geog_id in geog_ids:
        # print geog_id
        make_dummy_variable(geog_var, geog_id)
# TODO: Create ln version of these variable definitions..

