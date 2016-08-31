import numpy as np
import pandas as pd

import orca
import psycopg2
import pandas.io.sql as sql

import utils
import datasources
import variables

geography_base_id = orca.get_injectable('geography_id')


def db_to_df(conn_info, query):
    """Executes SQL query and returns DataFrame of results."""
    conn = psycopg2.connect(conn_info)
    return sql.read_sql(query, conn)

@orca.step('load_refinements')
def load_refinements(db_connection):
    scenario = orca.get_injectable('scenario') if 'scenario' in orca.list_injectables() else 'baseline'
    query = "select * from public.refinements"
    refinements = db_to_df(db_connection, query)
    orca.add_table('refinements', refinements)

@orca.step('process_scheduled_development_events')
def process_scheduled_development_events(zones, jobs, residential_units, households, db_connection):
    
    #check current scenario, if not specified, set to baseline
    scenario = orca.get_injectable('scenario') if 'scenario' in orca.list_injectables() else 'baseline'
    query = "select * from public.developmentprojects_block"
    devproj = db_to_df(db_connection, query)
    devproj = devproj[devproj.scenario==scenario]
    
    b = zones.to_frame(['square_meters_land','residential_unit_capacity','employment_capacity'])
    j = jobs.local
    ru = residential_units.local
    
    #split multi-block projects' units and emp capacity across those zones
    devproj_block = pd.DataFrame(columns=devproj.columns)
    block_list = []
    for i, row in devproj.iterrows():
        zones = np.arange(len(row.project_blocks))
        block_list.extend(row.project_blocks)
        for num in zones:
            devproj_block = devproj_block.append(dict(row), ignore_index=True)
    devproj_block['block'] = block_list
    
    #join res_units and jobs back to devproj for remaining capacity calcs
    devproj_block = pd.merge(devproj_block, b, how='left', left_on='block', right_index=True, suffixes=('','_existing'))
    ru = ru.groupby(geography_base_id).count().rename(columns={'year_built':'res_units_existing'}).drop('building_type_id', axis=1)
    devproj_block = pd.merge(devproj_block, ru, how='left', left_on='block', right_index=True)
    j = j.groupby(geography_base_id).count().rename(columns={'sector_id':'jobs_existing'})
    devproj_block = pd.merge(devproj_block, j, how='left', left_on='block', right_index=True)
    
    #add redevelopments to registered tables
    redevelopments = devproj_block[devproj_block.redevelopment_flag==True]
    redevelopments = redevelopments[['start_date','block']]
    if len(redevelopments.index) > 0:
        orca.add_table('redevelopments', redevelopments)
    
    #calculate total area, capacities within multi-block devprojs
    devproj_block['remaining_res_capacity'] = devproj_block.residential_unit_capacity - devproj_block.res_units_existing.fillna(0)
    devproj_block = pd.merge(devproj_block, 
         devproj_block.groupby('devproj_id').agg({'square_meters_land':'sum',
                                                  'remaining_res_capacity':'sum',
                                                  'residential_unit_capacity':'sum',
                                                  'employment_capacity_existing':'sum'}),
         how='left', left_on='devproj_id', right_index=True, suffixes=('','_total'))
    
    #create proportions for allocating units and capacity
    devproj_block['prop_area'] = np.divide(devproj_block.square_meters_land, devproj_block.square_meters_land_total.astype('float'))
    devproj_block['prop_ru'] = np.divide(devproj_block.residential_unit_capacity, devproj_block.residential_unit_capacity_total.astype('float'))
    devproj_block['prop_ru_cap'] = np.divide(devproj_block.remaining_res_capacity, devproj_block.remaining_res_capacity_total.astype('float'))
    devproj_block['prop_emp'] = np.divide(devproj_block.employment_capacity_existing,
                                          devproj_block.employment_capacity_existing_total.astype('float'))
    
    #split projects based on years
    devproj_block = devproj_block.ix[devproj_block.index.repeat(devproj_block.duration.astype('int'))]
    devproj_block['residential_units'] = ((devproj_block.residential_units * devproj_block.prop_ru_cap) / devproj_block.duration).round().astype('int')
    devproj_block['employment_capacity'] = ((devproj_block.employment_capacity * devproj_block.prop_emp) / devproj_block.duration).round().astype('int')

    years_to_add = []
    proj = []
    for i, row in devproj_block.iterrows():
        if i not in proj:
            years = np.arange(row.duration)
            years_to_add.extend(years)
            proj.append(i)
        else:
            pass
    devproj_block['year_built'] = devproj_block.start_date + years_to_add
    
    #separate res and nonres projects
    devproj_nonres = devproj_block[devproj_block.building_type_id==5]
    devproj_nonres = devproj_nonres[['block','year_built','employment_capacity']]
    devproj_block = devproj_block[devproj_block.building_type_id<5]
    
    #create individual unit records 
    devproj_block = devproj_block.reset_index()
    devproj_block = devproj_block.ix[devproj_block.index.repeat(devproj_block.residential_units.astype('int'))]
    devproj_block = devproj_block[['block','year_built','building_type_id']]
    
    orca.add_table('nonres_developments', devproj_nonres)
    orca.add_table('residential_developments', devproj_block) 
