import os
import sys
import time
import redis
import datetime
import argparse
import numpy as np
import pandas as pd

try:
    from google.cloud import datastore
    client = datastore.Client()
except:
    print 'No connection to Cloud Data Store available.'

import orca
import urbansim
from urbansim.models.regression import RegressionModel

import datasources
import variables
import models

np.random.seed(0)


def wait_for_key(key):
    print 'Waiting for %s.' % key

    while True:
        if r.exists(key) == True:
            break
        time.sleep(.1)


def process_queue(queue_key, process_func, items_name, eval_items=False):
    i = 0
    work_to_do = True
    results = {}

    while work_to_do:
        task = r.rpop(queue_key)
        if task is None:
            work_to_do = False
        else:
            if eval_items:
                task = eval(task)
            result = process_func(task)
            results[str(task)] = result
            i += 1

    print 'Processed %s %s.' % (i, items_name)
    return results


def process_queue_indefinitely(queue_key, process_func, items_name, eval_items=False):
    i = 0
    while True:
        task = r.rpop(queue_key)
        if task is None:
            print 'Waiting for tasks.  %s processed' % i
            time.sleep(.1)
            if r.get('specification_complete') == '1':
                break
        else:
            if eval_items:
                task = eval(task)
            result = process_func(task)
            i += 1

    print 'Processed %s %s.' % (i, items_name)


def lcm_estimation_function(agents, alternatives, location_variable_name, choosers_fit_filter=None):
    """
    For a given set of agents and alternatives, generates a model fitting
    function that can take arbitrary specification.
    Parameters
    ----------
    agents : pandas.DataFrame
        Table describing the agents making choices, e.g. households.
    alternatives : pandas.DataFrame
        Table describing the things from which agents are choosing,
        e.g. blocks.
    current_choice : string
        Name of column in agents table identifying the currently 
        chosen alternative.
    choosers_fit_filter : string, optional
        Chooser fit filter to apply at the model object level.
    Returns
    -------
    estimate_model : function
        Estimation function for the agents/alternatives for testing
        multiple specifications.
    """
    num_alts_to_sample = len(alternatives) - 1
    if num_alts_to_sample > 50:
        num_alts_to_sample = 50

    if (len(agents) > 1000) & (num_alts_to_sample >= 50):
        print 'Generating estimation function with estimation_sample_size'
        if choosers_fit_filter:
            print 'Generating estimation function with choosers_fit_filter'
            def estimate_model(specification):
                dcm = urbansim.models.dcm.MNLDiscreteChoiceModel(model_expression=specification,
                                                                 sample_size=num_alts_to_sample,
                                                                 probability_mode='single_chooser',
                                                                 choice_mode='aggregate',
                                                                 estimation_sample_size=1000,
                                                                 choosers_fit_filters=choosers_fit_filter)
                dcm.fit(agents, alternatives, location_variable_name)
                return dcm
        else:
            def estimate_model(specification):
                dcm = urbansim.models.dcm.MNLDiscreteChoiceModel(model_expression=specification,
                                                                 sample_size=num_alts_to_sample,
                                                                 probability_mode='single_chooser',
                                                                 choice_mode='aggregate',
                                                                 estimation_sample_size=1000)
                dcm.fit(agents, alternatives, location_variable_name)
                return dcm
    else:
        if choosers_fit_filter:
            print 'Generating estimation function with choosers_fit_filter'
            def estimate_model(specification):
                dcm = urbansim.models.dcm.MNLDiscreteChoiceModel(model_expression=specification,
                                                                 sample_size=num_alts_to_sample,
                                                                 probability_mode='single_chooser',
                                                                 choice_mode='aggregate',
                                                                 choosers_fit_filters=choosers_fit_filter)
                dcm.fit(agents, alternatives, location_variable_name)
                return dcm
        else:
            def estimate_model(specification):
                dcm = urbansim.models.dcm.MNLDiscreteChoiceModel(model_expression=specification,
                                                                 sample_size=num_alts_to_sample,
                                                                 probability_mode='single_chooser',
                                                                 choice_mode='aggregate')
                dcm.fit(agents, alternatives, location_variable_name)
                return dcm
    return estimate_model


def rm_estimation_function(observations, dep_var, fit_filters=None, predict_filters=None):
    """
    For a given set of observations, generates a regression model fitting
    function that can take arbitrary specification.

    Parameters
    ----------
    observations : pandas.DataFrame
        Table of observations to estimate model with.
    dep_var : str
        Name of the dependent variable, e.g. 'rent'
    fit_filter : string, optional
        Fit filter to apply at the model object level.
    predict_filter : string, optional
        Predict filter to apply at the model object level.

    Returns
    -------
    estimate_model : function
        Estimation function for the observations for testing
        multiple specifications on the same dataset/config.

    """
    def estimate_model(specification):
        rm = RegressionModel(model_expression={'left_side':dep_var, 'right_side':specification},
                             fit_filters = fit_filters,
                             predict_filters=predict_filters,
                             ytransform = np.exp)
        try:
            rm.fit(observations)
        except:
            return None

        return rm
    
    return estimate_model


def estimation_setup(alts, alternatives_id_name="block_id", store=None, store_results=False):
    # Wait for manager to provide name of estimation dataset table
    wait_for_key('estimation_dataset_type')
    estimation_dataset_type = r.get('estimation_dataset_type')

    wait_for_key('estimation_model_type')
    estimation_model_type = r.get('estimation_model_type')

    region = r.get('region')
    run_id = r.get('run_id')

    ## Get model chooser arguments and calculate choosers table
    
    if estimation_dataset_type == 'url':
        wait_for_key('estimation_dataset_url')
        estimation_dataset_url = r.get('estimation_dataset_url')
        agents_for_estimation = pd.read_csv(estimation_dataset_url,
                        dtype={
                            "block_id": "object"
                        }).set_index('lead_id')
        choosers_fit_filter = None
    else:
        model_name = r.get('model_name')
        if estimation_model_type == 'location':
            agents_name = r.get('agents_name')
            choosers_fit_filter = r.get('choosers_fit_filter')
            segmentation_variable = r.get('segmentation_variable')
            segment_id = int(r.get('segment_id'))
            if estimation_dataset_type == 'h5':
                agents_for_estimation = store[agents_name]
            else:
                agents_for_estimation = orca.get_table(agents_name).to_frame()
            # Consider only choosers/observations in the segment
            agents_for_estimation = agents_for_estimation[agents_for_estimation[segmentation_variable] == segment_id]
        elif estimation_model_type == 'regression':
            observations_name = r.get('observations_name')
            dep_var = r.get('dep_var')
            segmentation_variable = r.get('segmentation_variable')
            segment_id = r.get('segment_id')
            fit_filters = eval(r.get('fit_filters'))
            var_filter_terms = r.get('var_filter_terms')
            observations = alts # Temporarily to avoid redundant calcs
        else:
            print 'Estimation model type unrecognized'

    # Define the estimation function
    if estimation_model_type == 'location':
        if choosers_fit_filter == 'None':    choosers_fit_filter = None
        model_estim_fn = lcm_estimation_function(agents_for_estimation, alts, alternatives_id_name,
                                                      choosers_fit_filter=choosers_fit_filter)
        def estimate_model(spec):
            try:
                # Determine task type
                if type(spec) == tuple: # task with follow up action
                    action = spec[0]
                    spec_proposal = spec[1]
                elif type(spec) == list: # estimation task only
                    spec_proposal = spec

                # Estimate and record
                dcm = model_estim_fn(spec_proposal)
                print dcm.fit_parameters
                dcm.choosers_predict_filters = '%s == %s' % (segmentation_variable, segment_id)
                dcm.choice_column = alternatives_id_name
                llr = dcm.log_likelihoods['ratio']
                tscore = dcm.fit_parameters['T-Score'].to_dict()
                r.set(str(spec), (llr, tscore))

                yaml_str = dcm.to_yaml()
                if store_results:
                    # Persist estimation results to Google cloud datastore
                    #import pdb; pdb.set_trace()
                    ds_spec = datastore.Entity(key=client.key('Spec'), exclude_from_indexes=('yaml',))
                    ds_spec['created'] = datetime.datetime.utcnow()
                    ds_spec['creator'] = u'autospec'
                    ds_spec['region'] = unicode(region)
                    ds_spec['run_id'] = unicode(run_id)
                    ds_spec['model_type'] = unicode(estimation_model_type)
                    ds_spec['model_name'] = unicode(model_name)
                    ds_spec['agents'] = unicode(agents_name)
                    ds_spec['choosers_fit_filter'] = unicode(choosers_fit_filter)
                    ds_spec['segmentation'] = unicode(segmentation_variable)
                    ds_spec['segment_id'] = segment_id

                    variables = spec if type(spec) is list else spec[1]
                    ds_spec['variables'] = [unicode(variable) for variable in variables]
                    ds_spec['llr'] = llr
                    ds_tscores = datastore.Entity(key=client.key('Tscores'))
                    ds_tscores.update(tscore)
                    ds_spec['tscores']=ds_tscores
                    ds_spec['yaml'] = unicode(yaml_str)
                    try:
                        client.put(ds_spec)
                    except:
                        try:
                            time.sleep(1); print 'Trying datastore write failed, trying again in 1 second..'
                            client.put(ds_spec)
                        except:
                            import pdb; pdb.set_trace()

                # Optional follow up action
                if type(spec) == tuple:
                    if action == 'yaml persist':
                        r.set('yaml_' + model_name, yaml_str)

                r.incr('spec_processed_counter')
                return llr, tscore

            except:
                print 'Failed!'
                r.lpush('failed_spec_proposals', str(spec))
                #import pdb; pdb.set_trace()
                r.incr('spec_processed_counter')
                return None


    elif estimation_model_type == 'regression':
        model_estim_fn = rm_estimation_function(observations, dep_var, fit_filters=fit_filters)
        def estimate_model(spec):
            try:
                # Determine task type
                if type(spec) == tuple: # task with follow up action
                    action = spec[0]
                    spec_proposal = spec[1]
                elif type(spec) == list: # estimation task only
                    spec_proposal = spec

                # Estimate and record
                rm = model_estim_fn(spec_proposal)
                print rm.fit_parameters
                r2 = rm.model_fit.rsquared_adj
                tscore = rm.fit_parameters['T-Score'].to_dict()
                r.set(str(spec), (r2, tscore))

                # Optional follow up action
                if type(spec) == tuple:
                    if action == 'yaml persist':
                        rm.predict_filters = '%s == %s' % (segmentation_variable, segment_id)
                        yaml_str = rm.to_yaml()
                        r.set('yaml_' + model_name, yaml_str)

                r.incr('spec_processed_counter')
                return r2, tscore

            except:
                print 'Failed!'

                r.incr('spec_processed_counter')
                r.lpush('failed_spec_proposals', str(spec))
                return None


    return estimate_model


def process_spec_proposal_queue(estimate_function):
    wait_for_key('spec_proposal_queue')
    results = process_queue_indefinitely('spec_proposal_queue', estimate_function, 'spec proposals', eval_items=True)


def autospec_worker(redis_host, redis_port, geography='blocks', alternatives_id_name='block_id', build_network=True, 
                    data_path=None, kill_upon_complete=False, store_results=False):

    # Redis connection
    global r
    r = redis.Redis(redis_host, redis_port)

    # Pre-processing
    if data_path:
        store = pd.HDFStore(data_path)
        alts = store[geography]

    else:
        store = None

        orca.set_redis_connection(redis_host, redis_port)
        blocks = orca.get_table(geography)
        if build_network:
            orca.run(['build_networks'])

        # Wait for manager checkin
        wait_for_key('manager_checkedin')

        # Get all columns
        while True:
            if r.exists('columns') == True:
                columns = r.lrange('columns', 0, -1)
                orca.set_redis_columns(columns)
                break
            time.sleep(.1)

        # Process the column queue
        def calculate_variable(column_name):
            return blocks[column_name]
         
        column_results = process_queue('column_queue', calculate_variable, 'variables')
        wait_for_key('varcalc_checkedin') # Manager confirms that all columns processed

        ## Form the alternatives table
        print 'Creating the alternatives DataFrame.'
        alternatives_dict = {}
        for col in columns:  alternatives_dict[col] = blocks[col]
        alts = pd.DataFrame(alternatives_dict)

    ## Do work!
    while True:
        ##  Get agents and form the estimation function
        estimate_model = estimation_setup(alts, alternatives_id_name=alternatives_id_name, store=store, store_results=store_results)
        # Process the specification proposal queue
        process_spec_proposal_queue(estimate_model)
        time.sleep(.1)
        if kill_upon_complete:
            if r.get('specification_complete') == '1':
                break


# dcm = model_estim_fn(['block_groups_std_sector_id'])
# spec_proposals = r.lrange('spec_proposal_queue', 0, -1)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        time.sleep(2)

        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--redis_host", type=str, help="redis host")
        parser.add_argument("-p", "--redis_port", type=int, help="redis port")
        parser.add_argument("-t", "--template", type=str, help="model template")
        parser.add_argument("-d", "--data_file", help="path to .h5 data file")
        parser.add_argument("-k", "--kill_upon_complete", action="store_true", help="whether to use calibrated coeffs")
        parser.add_argument("-s", "--store_results", action="store_true", help="whether to persist estimation results")

        args = parser.parse_args()

        kill_upon_complete = True if args.kill_upon_complete else False
        store_results = args.store_results if args.store_results else False
        data_path = args.data_file if args.data_file else None

        if args.template == 'zone':
            build_network = False
            geography = 'zones'
            alternatives_id_name = 'zone_id'
        elif args.template == 'parcel':
            build_network = True
            geography = 'buildings'
            alternatives_id_name = 'building_id'
        else:
            build_network = True
            geography = 'blocks'
            alternatives_id_name = 'block_id'

        autospec_worker(args.redis_host, args.redis_port, geography=geography, build_network=build_network,
                        alternatives_id_name=alternatives_id_name, data_path=data_path, kill_upon_complete=kill_upon_complete,
                        store_results=store_results)

    else:
        print 'Need to specify redis host and port'
