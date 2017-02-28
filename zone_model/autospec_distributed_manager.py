import os
import sys
import time
import redis
import argparse
import numpy as np
import pandas as pd

import orca
import urbansim

import datasources
import variables
import models
import autospec_recipes

np.random.seed(0)


def wait_for_key(key):
    """
    Wait for Redis key to exist.

    Parameters
    ----------
    key : str
        The key to wait for the existence of.

    Returns
    -------
    None
    """
    print 'Waiting for %s.' % key

    while True:
        if r.exists(key) == True:
            break
        time.sleep(.1)


def monitor_queue(queue_key):
    """
    Wait for Redis queue to empty out (reach length zero).

    Parameters
    ----------
    queue_key : str
        The queue to monitor.

    Returns
    -------
    None
    """
    while True:
        time.sleep(.1)
        len_var_queue = r.llen(queue_key)
        if len_var_queue == 0:
            break


def connect_to_redis(redis_host, redis_port, flush=False):
    """
    Establish Redis connection and optionally flush existing keys.

    Parameters
    ----------
    redis_host : str
        Host of Redis to connect to.

    redis_port : int
        Host port number that Redis is listening on.

    flush : bool
        Whether or not to flush existing keys upon establishing connection.

    Returns
    -------
    r : Redis connection
        The Redis connection object to use for communication with the key-value store.
    """
    global r
    r = redis.Redis(redis_host, redis_port)
    orca.set_redis_connection(redis_host, redis_port)
    if flush:
        r.flushall()
    return r


def data_setup(core_table_name, build_network=True):
    """
    Calculate all variables in the variable library.  This is the feature set we will select
    from when specifying the model.  Allows for distributed calculation of variables (distributed
    among the autospec worker processes).

    Parameters
    ----------
    core_table_name : str
        Name of the core orca table to use in estimation. This is usually the geographic 
        level of the model (e.g. blocks) or the alternative set/estimation dataset.

    build_networks : boolean
        Whether or not to build pandana network.

    Returns
    -------
    core_table : Orca DataFrame wrapper
        DataFrame wrapper for the core table, used to access the underlying variables.
    """
    # Data pre-processing
    core_table = orca.get_table(core_table_name)
    columns = core_table.columns
    if build_network:
        orca.run(['build_networks'])

    # Populate column queue for workers to do variable calculations
    r.rpush('columns', *columns)
    r.rpush('column_queue', *columns)
    orca.set_redis_columns(columns)
    print 'Column queue populated.'

    # Monitor column queue processing
    monitor_queue('column_queue')
    print 'Variable calculations complete.'
    r.set('varcalc_checkedin', 1)
    
    return core_table


def define_estimation_function(model_name=None, agents_name=None, choosers_fit_filter=None, 
                               segmentation_variable=None, segment_id=None, estimation_dataset_type = 'urbansim',
                               estimation_model_type='location', url=None, observations_name=None,
                               dep_var=None, fit_filters=None, var_filter_terms=None):
    """
    Configure model estimation settings, e.g. which agents, filters, segmentation etc.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'hlcm1'.

    agents_name : str
        Name of the choosers in a LCM, e.g. 'households'.

    choosers_fit_filter : str
        Filter on choosers, e.g. 'recent_mover == 1'.

    segmentation_variable : str
        Variable to segment choosers on, e.g. 'income_quartile'.

    segment_id : int or str
        Value of the segmentation variable for inclusion in model, e.g. 1.

    estimation_dataset_type : str
        Type of dataset being used in estimation-  either 'urbansim' or 'url'.

    estimation_model_type : str
        Type of model to auto-specify-  either 'location' or 'regression'.

    url : str
        URL to estimation dataset, if estimation_dataset_type is 'url'.

    observations_name : str
        Name of table to estimate regression model with, e.g. 'blocks'.

    dep_var : str
        Name of column on observations table to be used as dependant variable in regression, e.g. 'res_rents'.

    fit_filters : list of str
        Estimation filters for regression model, e.g. ['res_rents > 0', 'rent_impute==0'].

    var_filter_terms : list of str
        Filter on explanatory variables-  which strings to filter on, e.g. ['value', 'rent'].

    Returns
    -------
    None
    """
    r.set('estimation_dataset_type', estimation_dataset_type)
    r.set('estimation_model_type', estimation_model_type)

    if estimation_dataset_type == 'url':
        #wait_for_key('estimation_dataset_url')
        #estimation_dataset_url = r.get('estimation_dataset_url')
        r.set('estimation_dataset_url', url)

    else:
        if estimation_model_type == 'location':
            r.set('model_name', model_name)
            r.set('agents_name', agents_name)
            r.set('choosers_fit_filter', choosers_fit_filter)
            r.set('segmentation_variable', segmentation_variable)
            r.set('segment_id', segment_id)

        elif estimation_model_type == 'regression':
            r.set('model_name', model_name)
            r.set('observations_name', observations_name)
            r.set('dep_var', dep_var)
            r.set('segmentation_variable', segmentation_variable)
            r.set('segment_id', segment_id)
            r.set('fit_filters', fit_filters)
            r.set('var_filter_terms', var_filter_terms)

        else:
            print 'Estimation model type unrecognized'


def subset_df_numeric_cols(df):
    """Returns DataFrame with only numeric columns."""
    dtypes = df.dtypes.reset_index().rename(columns={'index':'col', 0:'dtype'})
    numeric_cols = [alt[0] for alt in zip(dtypes.col, dtypes.dtype) if alt[1] != 'object'] # Only keep numeric columns
    df = df[numeric_cols]
    return df


def create_alternatives_dataframe(table):
    """Create alternatives DataFrame by fetching all calculated columns from Redis."""
    time.sleep(3) # look into why some column keys not found in multi-worker context if i don't sleep here
    print 'Creating the alternatives DataFrame.'
    alternatives_dict = {}
    for col in table.columns:  
        alternatives_dict[col] = table[col]  # This should draw from the distributed redis variable cache
    alts = pd.DataFrame(alternatives_dict)

    ## Consider numeric columns in estimation only
    alts = subset_df_numeric_cols(alts)

    return alts


def reset_specification_queue_status():
    """Reset to zero Redis keys indicating specification status."""
    time.sleep(1)
    r.set('specification_started', 0)
    r.set('specification_complete', 0)


def set_specification_status_complete():
    """Set Redis keys to indicate specification process complete."""
    r.delete('estimation_dataset_type') # Delete now until it is populated again by next model
    r.set('specification_complete', 1) # Instruct workers to break from their current listening loop


class AutospecManager(object):
    """
    Manager for the auto-specification process.  Coordinates data setup, step-wise model 
    specification routine, and final model selection.

    """
    def __init__(self, redis_host, redis_port, core_table_name='blocks', build_network=True):

        ## Redis connection
        connect_to_redis(redis_host, redis_port, flush=True)
        r.set('manager_checkedin', 1)
        print 'Redis connection made.'

        ## Data setup and initiate worker variable calculations
        core_table = data_setup(core_table_name, build_network=build_network)
        self.alts = create_alternatives_dataframe(core_table)


    def autospecify_urbansim_lcm_model(self, fitting_strategy='recipe', model_name='hlcm1', agents_name='households', 
                                   choosers_fit_filter='recent_mover == 1', segmentation_variable='income_quartile', segment_id=1,
                                   constraint_config=None, optimization_metric='significance'):

        if r.get('specification_complete') == '1':  
            reset_specification_queue_status() # If this is not the first spec run, reset spec status

        define_estimation_function(model_name=model_name, # function call parameters different if estimation dataset is url
                                   agents_name=agents_name, 
                                   choosers_fit_filter=choosers_fit_filter, 
                                   segmentation_variable=segmentation_variable,
                                   segment_id=segment_id, 
                                   estimation_model_type='location')

        ##  Auto-specify
        """evaluate_variable_set(alts.columns, [], 'lcm')"""
        if fitting_strategy == 'stepwise_simple':
            r.set('specification_started', 1)
            specification = stepwise([], self.alts.columns[:100], max_iterations=3, optimization_metric=optimization_metric)
            set_specification_status_complete()

        elif fitting_strategy == 'recipe':
            explanatory_variables = [var for vars in [category['variables']  for category in constraint_config] for var in vars]
            variable_pool = set(explanatory_variables)
            r.set('specification_started', 1)
            max_iterations = 0
            specification = apply_model_logic(variable_pool, max_iterations, constraint_config)

            # Get YAML string for final model
            specification_check = ('yaml persist', list(specification))
            r.rpush('spec_proposal_queue', specification_check)
            result_key = 'yaml_' + model_name
            wait_for_key(result_key)
            final_model = r.get(result_key)
            write_to_file(final_model, model_name + '.yaml')
            set_specification_status_complete()

        else:
            print 'Fitting strategy not recognized!'


    def autospecify_url_dataset_lcm(self, url_from_cmd):

        if r.get('specification_complete') == '1':  
            reset_specification_queue_status() # If this is not the first spec run, reset spec status

        r.set('specification_started', 1)
        wait_for_key('remote_estimation_dataset_url')   #### THIS LINE FOR DEMO ONLY-  TODO DELETE
        url = r.get('remote_estimation_dataset_url')    #### THIS LINE FOR DEMO ONLY- TODO DELETE
        define_estimation_function(estimation_dataset_type = 'url', url=url)
        specification = stepwise([], self.alts.columns, max_iterations = 10)
        set_specification_status_complete()


    def autospecify_urbansim_rm_model(self, fitting_strategy='recipe', model_name='repm1', observations_name='blocks', 
                                   dep_var='res_rents', segmentation_variable='all_blocks', segment_id=1, fit_filters=['res_rents > 0', 'rent_impute==0'],
                                   var_filter_terms=['value', 'rent'], constraint_config=None, apply_ln=True):

        if r.get('specification_complete') == '1':  
            reset_specification_queue_status() # If this is not the first spec run, reset spec status
        
        if apply_ln == True:
            dep_var = 'np.log1p(%s)' % dep_var
            print 'Dependent variables: %s' % dep_var

        define_estimation_function(model_name=model_name, # function call parameters different if estimation dataset is url
                                   observations_name=observations_name, 
                                   dep_var=dep_var, 
                                   segmentation_variable=segmentation_variable,
                                   segment_id=segment_id,
                                   fit_filters=fit_filters,
                                   var_filter_terms=var_filter_terms,
                                   estimation_model_type='regression')
        
        variable_pool = [var for var in self.alts.columns if (dep_var not in var) & (segmentation_variable not in var)] # Filter out vars based on dependent variable
        if var_filter_terms: # Filter out other vars from variable_pool if var_filter_terms specified
            for filter_var in var_filter_terms:
                variable_pool = [var for var in variable_pool if filter_var not in var]

        ##  Auto-specify
        if fitting_strategy == 'stepwise_simple':
            r.set('specification_started', 1)
            specification = rm_stepwise([], variable_pool, dep_var, max_iterations = 3)
            set_specification_status_complete()

        elif fitting_strategy == 'recipe':
            explanatory_variables = [var for vars in [category['variables']  for category in constraint_config] for var in vars]
            variable_pool = set(explanatory_variables)
            r.set('specification_started', 1)
            max_iterations = 0
            specification = rm_apply_model_logic(variable_pool, max_iterations, dep_var, constraint_config)

            # Get YAML string for final model
            specification_check = ('yaml persist', list(specification))
            r.rpush('spec_proposal_queue', specification_check)
            result_key = 'yaml_' + model_name
            wait_for_key(result_key)
            final_model = r.get(result_key)
            write_to_file(final_model, model_name + '.yaml')
            set_specification_status_complete()

        else:
            print 'Fitting strategy not recognized!'


def write_to_file(str_contents, file_path):
    yaml_file = open(file_path, "w")
    yaml_file.write(str_contents)
    yaml_file.close()


def wait_for_all_specs_processed(number_of_specs):
    print 'Waiting for all specs to finish being processed.'
    while True:
        time.sleep(.1)
        specs_processed = r.get('spec_processed_counter')
        if specs_processed == str(number_of_specs):
            print '    Estimation of specs in current queue complete.'
            time.sleep(.1)
            break


def submit_spec_jobs(variables_to_try, specification_proposals):
    r.set('spec_processed_counter', 0)

    ## Populate the specification proposal queue
    print 'Specification proposals to redis.'
    r.rpush('spec_proposal_queue', *specification_proposals)

    ## Wait until specification proposal queue processed.  When empty, model estimation is complete for this iteration.
    monitor_queue('spec_proposal_queue')
    number_of_specs = len(specification_proposals)
    wait_for_all_specs_processed(number_of_specs)
    
    # Get estimation results
    print 'Retrieving estimation results.'
    results = [r.get(str(spec_proposal)) for spec_proposal in specification_proposals]

    # Filter out models that errored-out
    print 'Filtering out invalid results'
    results = zip(variables_to_try, results)
    results = [model for model in results if (model[1] is not None)]
    models = [model for model in results if ('nan' not in model[1])]

    return models


def evaluate_variable_set(variable_pool, base_specification, model_type, variables_to_monitor=None, dep_var=None): 
    """
    For a given set of variables and a base specification, estimate the model
    with each new variable singly added to the specification in turn.  So if 
    the variable pool is ['b', 'c'] and the base specification is ['a'], then 
    the following specifications will be estimated:  ['a', 'b'],  ['a', 'c'].  

    Parameters
    ----------
    variable_pool : iterable of strings
        A set of variable names to try adding to the base specification.
    base_specification : iterable of strings
        A set of variable names that are currently in the base specification.
    model_type : str
        Can be 'lcm' or 'rm'
    variables_to_monitor : list of str
        Optional list of variables to track the coefficient/significance of.
    dep_var : str
        Only needed for regression models.

    Returns
    -------
    fit_measures : dictionary
        Dictionary of fit measures (e.g. log-likelihood ratios, adjusted r2) 
        where the key is variable name and the value is the log-likelihood ratio 
        achieved by adding this variable to the base specification.
    t_scores : dictionary
        Dictionary of t-values where the key is variable name and the value is 
        the t-value of the variable when added to the base specification.

    """
    variables_to_try = list(variable_pool)
    # Generate the new specification proposals to try
    print 'Generating specification proposals.'
    specification_proposals = []
    for variable in variables_to_try:
        specification_proposal = list(base_specification)
        specification_proposal.append(variable)
        specification_proposals.append(specification_proposal)

    ## Populate/process the specification proposal queue and retrieve/filter results
    models  = submit_spec_jobs(variables_to_try, specification_proposals)
    if len(models) == 0:
        print 'Zero successful models returned by workers-  trying again in 5 seconds..'
        time.sleep(5)
        models  = submit_spec_jobs(variables_to_try, specification_proposals)
        if len(models) == 0:
            print 'Retry yielded zero successful models.'
            red = r
            import pdb; pdb.set_trace()

    # Get the updated variables to try (post-filter: specifications that were successfully estimated)
    variables_to_try, models = zip(*models)
    models = [eval(model) for model in models]

    # Record metrics across specification proposals
    print 'Recording fit and significance metrics'
    fit_measures = [model[0] for model in models]

    models = zip(variables_to_try, models)

    t_scores = [model[1][1][model[0]] for model in models]

    if variables_to_monitor:
        tscores_monitor = {}
        for var in variables_to_monitor:
            var_tscores = [model[1][1][var] for model in models]
            var_tscores = dict(zip(variables_to_try, var_tscores))
            tscores_monitor[var] = var_tscores

    fit_measures = dict(zip(variables_to_try, fit_measures))
    t_scores = dict(zip(variables_to_try, t_scores))
    models = dict(models)

    if variables_to_monitor:
        return fit_measures, t_scores, models, tscores_monitor
    else:
        return fit_measures, t_scores, models
        

def stepwise(base_specification, variable_pool, minimum_t_value=1.8, optimization_metric='significance',
             count=1, allow_backward_steps=True, max_iterations = 100, vars_to_monitor=None):
    """
    Step-wise feature selection routine for automatically building out model specifications.

    Parameters
    ----------
    base_specification : iterable of strings
        A set of variable names to use as a base when starting the stepwise 
        routine, if any. An empty set is acceptable.
    variable_pool : iterable of strings
        The pool of explanatory variables to try in the specifications.  This 
        is the universe of possible features to be considered.
    minimum_t_value : float, optional
        Minimum variable t-value that is acceptable.  Below this value and 
        variable will be rejected outright.
    optimization_metric : string, optional
        Either 'significance' or 'fit', defaults to 'significance'
    count : integer, optional
        Start-value of count variable to keep track of iteration number.
    allow_backward_steps : boolean, optional
        True if backward steps allowed, else False.  Backward steps remove 
        variables that lose their significance from the specification and
        return them to to the variable pool.
    max_iterations : integer, optional
        Maximum number of iterations of the stepwise routine.  Stop if the
        maximum is reached.
    vars_to_monitor : dict, optional
        Optional dictionary to map between specific category variables and
        required sign
    Returns
    -------
    specification : set
        Set of variable names in final specification.

    """
    if count > max_iterations:
        print 'Maximum iterations reached.  Ending stepwise.'
        return base_specification
        
    print '*Iteration %s'% count
    print base_specification
    # Evaluate variables in pool
    if vars_to_monitor:
        log_likelihood_ratios, t_scores, models, tscores_monitor = evaluate_variable_set(variable_pool,
                                                                   base_specification,'lcm',
                                                                   variables_to_monitor = vars_to_monitor.keys())
        if len(vars_to_monitor) > 0:
            t_scores =  pd.Series(t_scores)
            log_likelihood_ratios = pd.Series(log_likelihood_ratios)
            for var in vars_to_monitor.keys():
                tscores_var = pd.Series(tscores_monitor[var])

                if vars_to_monitor[var] == 'positive':
                    tscores_var = tscores_var[tscores_var > 1.96]
                if vars_to_monitor[var] == 'negative':
                    tscores_var = tscores_var[tscores_var < -1.96]
                else:
                    tscores_var = tscores_var[tscores_var.abs() > 1.96]
                t_scores = t_scores[np.in1d(t_scores.index, tscores_var.index)]
                log_likelihood_ratios = log_likelihood_ratios[np.in1d(log_likelihood_ratios.index, tscores_var.index)]
            t_scores = t_scores.to_dict()
            log_likelihood_ratios = log_likelihood_ratios.to_dict()

    else:
        log_likelihood_ratios, t_scores, models = evaluate_variable_set(variable_pool,
                                                                base_specification,
                                                                'lcm')

    log_likelihood_ratios = pd.Series(log_likelihood_ratios)
    t_scores = pd.Series(t_scores)
    model_metrics = pd.DataFrame({'llr':log_likelihood_ratios, 'ts':t_scores})
    model_metrics = model_metrics[model_metrics.llr != 0]  # Filter out LLR of 0 (likely a data problem)
    
    # Get max LLR and variable name associated with it
    idx_max_llr, max_llr = idx_of_max_value(model_metrics.llr)

    # Get max T-score and variable name associated with it
    idx_max_ts, max_ts = idx_of_max_value(model_metrics.ts.abs())
    
    # Get model associated with max t-score
    if optimization_metric == 'significance':
        print "Selecting top specification based on significance"
        latest_model = models[idx_max_ts]
    elif optimization_metric == 'fit':
        print "Selecting top specification based on fit"
        max_llr_model = models[idx_max_llr]
        tscore_of_max_llr_var = max_llr_model[1][idx_max_llr]
        if tscore_of_max_llr_var > minimum_t_value:
            latest_model = models[idx_max_llr]
            max_ts = tscore_of_max_llr_var
            idx_max_ts = idx_max_llr
        else:
            latest_model = models[idx_max_ts]

    print 'Log-likelihood is %s.' % latest_model[0]
    print 'T-scores are %s.' % latest_model[1]
    accepted_t_scores = pd.Series(latest_model[1])

    # Decision rule
    variable_pool = set(variable_pool)
    base_specification = set(base_specification)
    if max_ts > minimum_t_value:
        variable_pool.remove(idx_max_ts)
        print 'Addings %s to specification' % idx_max_ts
        base_specification.add(idx_max_ts)
        
        # Step backwards (remove vars that lose significance) if needed.  Put insignificant vars back in pool.
        if allow_backward_steps:
            insignificant_vars = accepted_t_scores[np.abs(accepted_t_scores) < minimum_t_value].index.values
            if len(insignificant_vars) > 0:
                for var in insignificant_vars:  
                    print '    Backward step:  removing %s from the specification' % var
                    base_specification.remove(var)
                    variable_pool.add(var)

        # Update redis with this iteration's chosen specification
        r.set('iteration_%s_spec' % count, base_specification)
        r.set('iteration_%s_model' % count, latest_model)
        time.sleep(.2)

        if vars_to_monitor:
            return stepwise(base_specification, variable_pool, count=count+1, max_iterations=max_iterations,
                            vars_to_monitor=vars_to_monitor, optimization_metric=optimization_metric)
        else:
            return stepwise(base_specification, variable_pool, count=count+1, max_iterations=max_iterations, 
                            optimization_metric=optimization_metric)
        
    else:
        print 'Specification complete in %s iterations' % count
        return base_specification


def rm_stepwise(base_specification, variable_pool, dep_var, minimum_t_value=1.8, count=1, allow_backward_steps=True,
             max_iterations = 100, vars_to_monitor=None):
    """
    Step-wise feature selection routine for automatically building out regression model specifications.

    Parameters
    ----------
    base_specification : iterable of strings
        A set of variable names to use as a base when starting the stepwise 
        routine, if any. An empty set is acceptable.
    variable_pool : iterable of strings
        The pool of explanatory variables to try in the specifications.  This 
        is the universe of possible features to be considered.
    minimum_t_value : float, optional
        Minimum variable t-value that is acceptable.  Below this value and 
        variable will be rejected outright.
    count : integer, optional
        Start-value of count variable to keep track of iteration number.
    allow_backward_steps : boolean, optional
        True if backward steps allowed, else False.  Backward steps remove 
        variables that lose their significance from the specification and
        return them to to the variable pool.
    max_iterations : integer, optional
        Maximum number of iterations of the stepwise routine.  Stop if the
        maximum is reached.
    vars_to_monitor : dict, optional
        Optional dictionary to map between specific category variables and
        required sign
    Returns
    -------
    specification : set
        Set of variable names in final specification.

    """
    if count > max_iterations:
        print 'Maximum iterations reached.  Ending stepwise.'
        return base_specification
        
    print '*Iteration %s'% count
    print base_specification
    # Evaluate variables in pool
    if vars_to_monitor:
        rsquareds, t_scores, models, tscores_monitor = evaluate_variable_set(variable_pool,
                                                                        base_specification,
                                                                        'rm', # model type
                                                                        variables_to_monitor = vars_to_monitor.keys(),
                                                                        dep_var = dep_var)
        if len(vars_to_monitor) > 0:
            t_scores =  pd.Series(t_scores)
            rsquareds = pd.Series(rsquareds)
            for var in vars_to_monitor.keys():
                tscores_var = pd.Series(tscores_monitor[var])

                if vars_to_monitor[var] == 'positive':
                    tscores_var = tscores_var[tscores_var > 1.96]
                if vars_to_monitor[var] == 'negative':
                    tscores_var = tscores_var[tscores_var < -1.96]
                else:
                    tscores_var = tscores_var[tscores_var.abs() > 1.96]
                t_scores = t_scores[np.in1d(t_scores.index, tscores_var.index)]
                rsquareds = rsquareds[np.in1d(rsquareds.index, tscores_var.index)]
            t_scores = t_scores.to_dict()
            rsquareds = rsquareds.to_dict()

    else:
        rsquareds, t_scores, models = evaluate_variable_set(variable_pool,
                                                            base_specification,
                                                            'rm', # model type
                                                            dep_var = dep_var)

    rsquareds = pd.Series(rsquareds)
    t_scores = pd.Series(t_scores)
    model_metrics = pd.DataFrame({'r2':rsquareds, 'ts':t_scores})
    model_metrics = model_metrics[(model_metrics.r2 > 0) & (model_metrics.r2 < 1)]  # Filter out R2 of 0.0, 1.0, or negative (likely a data problem)
        
    # Get max R2 and variable name associated with it
    idx_max_llr, max_llr = idx_of_max_value(model_metrics.r2)

    # Get max T-score and variable name associated with it
    idx_max_ts, max_ts = idx_of_max_value(model_metrics.ts.abs())
    
    # Get model associated with max t-score
    latest_model = models[idx_max_ts]
    print 'Adjusted R2 is %s.' % latest_model[0]
    print 'T-scores are %s.' % latest_model[1]
    accepted_t_scores = pd.Series(latest_model[1])

    # Decision rule
    variable_pool = set(variable_pool)
    base_specification = set(base_specification)
    if max_ts > minimum_t_value:
        variable_pool.remove(idx_max_ts)
        print 'Addings %s to specification' % idx_max_ts
        base_specification.add(idx_max_ts)
        
        # Step backwards (remove vars that lose significance) if needed.  Put insignificant vars back in pool.
        if allow_backward_steps:
            insignificant_vars = accepted_t_scores[np.abs(accepted_t_scores) < minimum_t_value].index.values
            if len(insignificant_vars) > 0:
                for var in insignificant_vars:  
                    print '    Backward step:  removing %s from the specification' % var
                    base_specification.remove(var)
                    variable_pool.add(var)

        # Update redis with this iteration's chosen specification
        r.set('iteration_%s_spec' % count, base_specification)
        r.set('iteration_%s_model' % count, latest_model)
        time.sleep(.2)

        if vars_to_monitor:
            return stepwise(base_specification, variable_pool, dep_var, count=count+1, max_iterations=max_iterations,
                            vars_to_monitor=vars_to_monitor)
        else:
            return stepwise(base_specification, variable_pool, dep_var, count=count+1, max_iterations=max_iterations)
        
    else:
        print 'Specification complete in %s iterations' % count
        return base_specification


def idx_of_max_value(series):
    """
    Returns maximum value of pd.Series and the index name corresponding to it.

    Parameters
    ----------
    series : pd.Series
        Series to find the maximum value in.

    Returns
    -------
    idx : string
        Index corresponding to the max value, typically a string.
    max value : float
        Maximum value, typically a float.

    """
    max_value = series.max()
    idx = series[series==max_value].index.values[0]
    return idx, max_value

    #print 'Flushing Redis keys'
    #r.flushall()
    #time.sleep(1)
    

    ### Do autospec logic here
    #    pass


def evaluate_variable_category(variable_group, required_sign, vars_to_monitor, base_specification):

    if len(vars_to_monitor) > 0:
        log_likelihood_ratios, t_scores, models, tscores_monitor = evaluate_variable_set(variable_group,
                                                                    base_specification,
                                                                    'lcm',
                                                                    variables_to_monitor = vars_to_monitor.keys())
    else:
        log_likelihood_ratios, t_scores, models = evaluate_variable_set(variable_group,
                                                                    base_specification,
                                                                    'lcm')

    t_scores =  pd.Series(t_scores)
    log_likelihood_ratios = pd.Series(log_likelihood_ratios)

    if required_sign == 'positive':
        significant_t_scores = t_scores[t_scores > 1.96] # Enforce positive sign and significance
    elif required_sign == 'negative':
        significant_t_scores = t_scores[t_scores < -1.96] # Enforce positive sign and significance
    else:
        significant_t_scores = t_scores[t_scores.abs() > 1.96] # Enforce significance only

    if len(significant_t_scores) > 0:
        sign_log_likelihood_ratios = log_likelihood_ratios[np.in1d(log_likelihood_ratios.index, significant_t_scores.index)]

        if len(vars_to_monitor) > 0:
            for var in vars_to_monitor.keys():
                tscores_var = pd.Series(tscores_monitor[var])

                if vars_to_monitor[var] == 'positive':
                    tscores_var = tscores_var[tscores_var > 1.96]
                if vars_to_monitor[var] == 'negative':
                    tscores_var = tscores_var[tscores_var < -1.96]
                else:
                    tscores_var = tscores_var[tscores_var.abs() > 1.96]
                sign_log_likelihood_ratios = sign_log_likelihood_ratios[np.in1d(sign_log_likelihood_ratios.index, tscores_var.index)]

        if len(sign_log_likelihood_ratios) > 0:
            idx_max_llr, max_llr = idx_of_max_value(sign_log_likelihood_ratios)

            print idx_max_llr
            print max_llr

            base_specification.add(idx_max_llr)
            vars_to_monitor[idx_max_llr] = required_sign
        else:
            print 'No variable that does not counter-productively influence an existing required category'

    else:
        print 'No significant variables'

    return base_specification, vars_to_monitor


def apply_model_logic(variable_pool, max_iterations, constraints):

    def get_variables_for_category_name(constraints, category_name):
        for variable_category in constraints:
            if variable_category['name'] == category_name:
                return variable_category['variables']
            
    def get_sign_for_category_name(constraints, category_name):
        for variable_category in constraints:
            if variable_category['name'] == category_name:
                return variable_category['sign']

    all_categories = []
    required_categories = []
    secondary_categories = []
    for variable_category in constraints:
        all_categories.append(variable_category['variables'])
        if variable_category['required']:
            required_categories.append((variable_category['variables'], variable_category['sign']))
        else:
            secondary_categories.append((variable_category['name'], variable_category['variables'], variable_category['sign']))

    base_specification = set([])
    vars_to_monitor = {}

    # Required categories
    for variable_group, required_sign in required_categories:
        base_specification, vars_to_monitor  = evaluate_variable_category(variable_group,
                                                                          required_sign,
                                                                          vars_to_monitor,
                                                                          base_specification)

    # Secondary categories
    secondary_category_names = [secondary_category[0] for secondary_category in secondary_categories]
    categories_by_tier = pd.DataFrame([category_name.split('_tier') for category_name in secondary_category_names], columns=['category_name', 'tier'])
    categories_by_tier['tier'] = categories_by_tier.tier.astype('int32')

    distinct_category_buckets = np.unique(categories_by_tier.category_name)
    for category_bucket in distinct_category_buckets:
        secondary_category_group = categories_by_tier[categories_by_tier.category_name == category_bucket]
        min_tier = secondary_category_group.tier.min()
        max_tier = secondary_category_group.tier.max()
        tiers = range(min_tier, max_tier + 1)
        for tier in tiers:
            current_tier = category_bucket + '_tier%s'%tier
            print current_tier
            last_tier_run = tier - 1
            prior_tier_variables = [var for vars in [get_variables_for_category_name(constraints, category_bucket + '_tier%s'%prior_tier) for prior_tier in np.arange(last_tier_run) + 1] for var in vars]
            current_tier_variables = get_variables_for_category_name(constraints, current_tier)
            current_tier_sign = get_sign_for_category_name(constraints, current_tier)

            if len(base_specification.intersection(set(prior_tier_variables))) == 0:
                base_specification, vars_to_monitor  = evaluate_variable_category(current_tier_variables, #variable_group
                                                                                  current_tier_sign, #required_sign
                                                                                  vars_to_monitor,
                                                                                  base_specification)
                if len(base_specification.intersection(set(current_tier_variables))) > 0:
                    break

    # Remove categorized variables from the variable pool
    for variable_group in all_categories:
        variable_pool = variable_pool.difference(set(variable_group))

    # Run the generalized step-wise routine to add additional variables while keeping the chosen categorized variables
    specification = stepwise(base_specification, variable_pool, max_iterations = max_iterations,
                             vars_to_monitor=vars_to_monitor)

    return specification


def rm_apply_model_logic(variable_pool, max_iterations, dep_var, constraints):

    def get_variables_for_category_name(constraints, category_name):
        for variable_category in constraints:
            if variable_category['name'] == category_name:
                return variable_category['variables']
            
    def get_sign_for_category_name(constraints, category_name):
        for variable_category in constraints:
            if variable_category['name'] == category_name:
                return variable_category['sign']

    all_categories = []
    required_categories = []
    secondary_categories = []
    for variable_category in constraints:
        all_categories.append(variable_category['variables'])
        if variable_category['required']:
            required_categories.append((variable_category['variables'], variable_category['sign']))
        else:
            secondary_categories.append((variable_category['name'], variable_category['variables'], variable_category['sign']))

    base_specification = set([])
    vars_to_monitor = {}

    # Required categories
    for variable_group, required_sign in required_categories:
        base_specification, vars_to_monitor  = rm_evaluate_variable_category(variable_group,
                                                                          required_sign,
                                                                          vars_to_monitor,
                                                                          base_specification, dep_var)

    # Secondary categories
    secondary_category_names = [secondary_category[0] for secondary_category in secondary_categories]
    categories_by_tier = pd.DataFrame([category_name.split('_tier') for category_name in secondary_category_names], columns=['category_name', 'tier'])
    categories_by_tier['tier'] = categories_by_tier.tier.astype('int32')

    distinct_category_buckets = np.unique(categories_by_tier.category_name)
    for category_bucket in distinct_category_buckets:
        secondary_category_group = categories_by_tier[categories_by_tier.category_name == category_bucket]
        min_tier = secondary_category_group.tier.min()
        max_tier = secondary_category_group.tier.max()
        tiers = range(min_tier, max_tier + 1)
        for tier in tiers:
            current_tier = category_bucket + '_tier%s'%tier
            print current_tier
            last_tier_run = tier - 1
            prior_tier_variables = [var for vars in [get_variables_for_category_name(constraints, category_bucket + '_tier%s'%prior_tier) for prior_tier in np.arange(last_tier_run) + 1] for var in vars]
            current_tier_variables = get_variables_for_category_name(constraints, current_tier)
            current_tier_sign = get_sign_for_category_name(constraints, current_tier)

            if len(base_specification.intersection(set(prior_tier_variables))) == 0:
                base_specification, vars_to_monitor  = rm_evaluate_variable_category(current_tier_variables, #variable_group
                                                                                  current_tier_sign, #required_sign
                                                                                  vars_to_monitor,
                                                                                  base_specification, dep_var)
                if len(base_specification.intersection(set(current_tier_variables))) > 0:
                    break

    # Remove categorized variables from the variable pool
    for variable_group in all_categories:
        variable_pool = variable_pool.difference(set(variable_group))

    # Run the generalized step-wise routine to add additional variables while keeping the chosen categorized variables
    if max_iterations > 0:
        specification = stepwise(base_specification, variable_pool, dep_var, max_iterations = max_iterations,
                                 vars_to_monitor=vars_to_monitor)
    else:
        specification = base_specification

    return specification


def rm_evaluate_variable_category(variable_group, required_sign, vars_to_monitor, base_specification, dep_var):

    if len(vars_to_monitor) > 0:
        rsquareds, t_scores, models, tscores_monitor = evaluate_variable_set(variable_group,
                                                                    base_specification,
                                                                    'rm', # model type
                                                                    variables_to_monitor = vars_to_monitor.keys(),
                                                                    dep_var = dep_var)
    else:
        rsquareds, t_scores, models = evaluate_variable_set(variable_group,
                                                                    base_specification,
                                                                    'rm', # model type
                                                                    dep_var = dep_var)

    t_scores =  pd.Series(t_scores)
    rsquareds = pd.Series(rsquareds)

    if required_sign == 'positive':
        significant_t_scores = t_scores[t_scores > 1.96] # Enforce positive sign and significance
    elif required_sign == 'negative':
        significant_t_scores = t_scores[t_scores < -1.96] # Enforce positive sign and significance
    else:
        significant_t_scores = t_scores[t_scores.abs() > 1.96] # Enforce significance only

    if len(significant_t_scores) > 0:
        sign_rsquareds = rsquareds[np.in1d(rsquareds.index, significant_t_scores.index)]

        if len(vars_to_monitor) > 0:
            for var in vars_to_monitor.keys():
                tscores_var = pd.Series(tscores_monitor[var])

                if vars_to_monitor[var] == 'positive':
                    tscores_var = tscores_var[tscores_var > 1.96]
                if vars_to_monitor[var] == 'negative':
                    tscores_var = tscores_var[tscores_var < -1.96]
                else:
                    tscores_var = tscores_var[tscores_var.abs() > 1.96]
                sign_rsquareds = sign_rsquareds[np.in1d(sign_rsquareds.index, tscores_var.index)]

        if len(sign_rsquareds) > 0:
            idx_max_r2, max_r2 = idx_of_max_value(sign_rsquareds)

            print idx_max_r2
            print max_r2

            base_specification.add(idx_max_r2)
            vars_to_monitor[idx_max_r2] = required_sign
        else:
            print 'No variable that does not counter-productively influence an existing required category'

    else:
        print 'No significant variables'

    return base_specification, vars_to_monitor


if __name__ == '__main__':

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--redis_host", type=str, help="redis host")
        parser.add_argument("-p", "--redis_port", type=int, help="redis port")
        args = parser.parse_args()

        #autospec_manager(args.redis_host, args.redis_port)
        autospec_manager = AutospecManager(args.redis_host, args.redis_port, core_table_name='zones', build_network=False)

        #autospec_manager.autospecify_url_dataset_lcm('http://synthpop-data2.s3-website-us-west-1.amazonaws.com/testdata/flint-lead-samples.csv')

        """
        autospec_manager.autospecify_urbansim_lcm_model(fitting_strategy='stepwise_simple', model_name='hlcm1', agents_name='households', 
                                   choosers_fit_filter='recent_mover == 1', segmentation_variable='income_quartile', segment_id=1, 
                                   optimization_metric='significance')
        """

        autospec_manager.autospecify_urbansim_lcm_model(fitting_strategy='stepwise_simple', model_name='hlcm1', agents_name='households', 
                                   choosers_fit_filter='None', segmentation_variable='income_quartile', segment_id=1, 
                                   optimization_metric='significance')
        
        #autospec_manager.autospecify_urbansim_rm_model(fitting_strategy='stepwise_simple')

        """
        autospec_manager.autospecify_urbansim_lcm_model(fitting_strategy='recipe', model_name='hlcm1', agents_name='households', 
                                   choosers_fit_filter='recent_mover == 1', segmentation_variable='income_quartile', segment_id=1)
        autospec_manager.autospecify_urbansim_lcm_model(fitting_strategy='recipe', model_name='hlcm2', agents_name='households', 
                                   choosers_fit_filter='recent_mover == 1', segmentation_variable='income_quartile', segment_id=2)
        autospec_manager.autospecify_urbansim_lcm_model(fitting_strategy='recipe', model_name='hlcm3', agents_name='households', 
                                   choosers_fit_filter='recent_mover == 1', segmentation_variable='income_quartile', segment_id=3)
        """
        #autospecify_urbansim_rm_model(self, fitting_strategy='recipe', model_name='repm1', observations_name='blocks', 
        #                           dep_var='res_rents', segmentation_variable='all_blocks', segment_id=1, fit_filters=['res_rents > 0', 'rent_impute==0'],
        #                           var_filter_terms=['value', 'rent'], constraint_config=None, apply_ln=True)




        #model = 'repm_rent'
        #autospec_manager.autospecify_urbansim_rm_model(fitting_strategy='recipe', model_name='repm1', observations_name='blocks', 
        #                           dep_var='res_rents', segmentation_variable='all_blocks', segment_id=1, fit_filters=['res_rents > 0', 'rent_impute==0'],
        #                           var_filter_terms=['value', 'rent'], constraint_config=constraint_configs[model + '_constraints.yaml'], )
        """
        model_structure = orca.get_injectable('model_structure')
        template_name = model_structure['template']
        models = model_structure['models']

        constraint_configs = orca.get_injectable('constraint_configs')[template_name]
        yaml_configs = {}
        for model in models.keys():
            print model
            constraint_config = constraint_configs[model + '_constraints.yaml']
            attribs = models[model]
            print attribs
            if attribs['model_type'] == 'location_choice':
                segment_ids = orca.get_table(attribs['agents_name'])[attribs['segmentation_variable']].value_counts().index.values
                for segment_id in segment_ids:
                    autospec_manager.autospecify_urbansim_lcm_model(fitting_strategy = 'recipe', 
                                                                    model_name = model + str(segment_id), 
                                                                    agents_name = attribs['agents_name'], 
                                                                    choosers_fit_filter = attribs['choosers_fit_filter'], 
                                                                    segmentation_variable = attribs['segmentation_variable'], 
                                                                    segment_id=segment_id,
                                                                    constraint_config=constraint_config)




            elif attribs['model_type'] == 'regression':
                segment_ids = orca.get_table(attribs['observations_name'])[attribs['segmentation_variable']].value_counts().index.values
                for segment_id in segment_ids:
                    autospec_manager.autospecify_urbansim_rm_model(fitting_strategy = 'recipe', 
                                                                    model_name = model + str(segment_id), 
                                                                    observations_name=attribs['observations_name'], 
                                                                    dep_var=attribs['dep_var'],
                                                                    segmentation_variable=attribs['segmentation_variable'], 
                                                                    segment_id=segment_id,
                                                                    fit_filters=attribs['fit_filters'],
                                                                    var_filter_terms=attribs['var_filter_terms'],
                                                                    constraint_config=constraint_config
                                                                    )
        """

        
        #autospec_manager.autospecify_url_dataset_lcm('http://synthpop-data2.s3-website-us-west-1.amazonaws.com/testdata/flint-lead-samples.csv')

    else:
        print 'Need to specify redis host and port'
