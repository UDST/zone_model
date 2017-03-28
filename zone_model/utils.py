import os
import yaml
import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc
from urbansim.models import GrowthRateTransition

import datasources


def default_choices(model, choosers, alternatives):
    """
    Simulate choices by just using MNLDiscreteChoiceModel's 
    predict function.
    Parameters
    ----------
    model : urbansim.models.MNLDiscreteChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    choices = model.predict(choosers, alternatives, debug=True)
    return choices


def random_choices(model, choosers, alternatives):
    """
    Simulate choices using random choice, weighted by probability
    but not capacity constrained.
    Parameters
    ----------
    model : urbansim.models.MNLDiscreteChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    probabilities = calculate_probabilities(model, choosers, alternatives)
    choices = np.random.choice(
        probabilities.index, size=len(choosers), replace=True, p=probabilities.values)
    return pd.Series(choices, index=choosers.index)



def simulate_choice_model(model, choice_function=default_choices, save_probabilities=False, **kwargs):
    """
    Computing choices, with arbitrary function for handling simulation strategy. 
    Parameters
    ----------
    model : urbansim.models.MNLDiscreteChoiceModel
        Fitted model object.
    choice_function : function
        Function defining how to simulate choices based on fitted model.
        Function must accept the following 3 arguments:  model object, choosers
        DataFrame, and alternatives DataFrame.  Additional optional keyword
        args can be utilized by function if needed (kwargs).
    save_probabilities : bool
        If true, will save the calculated probabilities underlying the simulation 
        as an orca injectable with name 'probabilities_modelname_itervar'.
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID. Some choosers
        will map to a nan value when there are not enough alternatives
        for all the choosers.
    """
    choosers, alternatives = calculate_model_variables(model)
    
    # By convention, choosers are denoted by a -1 value in the choice column
    choosers = choosers[choosers[model.choice_column] == -1]
    print "%s agents are making a choice." % len(choosers)
    
    choices = choice_function(model, choosers, alternatives, **kwargs)
    
    if save_probabilities:
        if not model.sim_pdf:
            probabilities = calculate_probabilities(model, choosers, alternatives)
        else:
            probabilities = model.sim_pdf.reset_index().set_index('alternative_id')[0]
        orca.add_injectable('probabilities_%s_%s' % (model.name, orca.get_injectable('iter_var')),
                            probabilities)
    
    return choices


def calculate_probabilities(model, choosers, alternatives):
    """
    Calculate model probabilities.
    Parameters
    ----------
    model : urbansim.models.MNLDiscreteChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    Returns
    -------
    probabilities : pandas.Series
        Mapping of alternative ID to probabilities.
    """
    probabilities = model.probabilities(choosers, alternatives)
    probabilities = probabilities.reset_index().set_index('alternative_id')[0] # remove chooser_id col from idx
    return probabilities


def calculate_model_variables(model):
    """
    Calculate variables needed to simulate the model, and returns DataFrames 
    of simulation-ready tables with needed variables.
    Parameters
    ----------
    model : urbansim.models.MNLDiscreteChoiceModel
        Model object with the following attributes:  choice_column, choosers, 
        alternatives.  Optional attributes are:  supply_variable, vacant_variable.
    Returns
    -------
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    """
    
    columns_used = model.columns_used() + [model.choice_column]
    choosers = orca.get_table(model.choosers).to_frame(columns_used)
    
    supply_column_names = [col for col in [model.supply_variable, model.vacant_variable] if col is not None]
    alternatives = orca.get_table(model.alternatives).to_frame(columns_used + list(supply_column_names))
    return choosers, alternatives


def unit_choices(model, choosers, alternatives):
    """
    Simulate choices using unit choice.  Alternatives table is expanded
    to be of length alternatives.vacant_variables, then choices are simulated
    from among the universe of vacant units, respecting alternative capacity.
    Parameters
    ----------
    model : urbansim.models.MNLDiscreteChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    
    supply_variable, vacant_variable = model.supply_variable, model.vacant_variable
    
    available_units = alternatives[supply_variable]
    vacant_units = alternatives[vacant_variable]
    vacant_units = vacant_units[vacant_units.index.values >= 0]  ## must have positive index 

    print "There are %d total available units" % available_units.sum()
    print "    and %d total choosers" % len(choosers)
    print "    but there are %d overfull buildings" % \
          len(vacant_units[vacant_units < 0])

    vacant_units = vacant_units[vacant_units > 0]

    indexes = np.repeat(vacant_units.index.values,
                        vacant_units.values.astype('int'))
    isin = pd.Series(indexes).isin(alternatives.index)
    missing = len(isin[isin == False])
    indexes = indexes[isin.values]
    units = alternatives.loc[indexes].reset_index()

    print "    for a total of %d temporarily empty units" % vacant_units.sum()
    print "    in %d buildings total in the region" % len(vacant_units)

    if missing > 0:
        print "WARNING: %d indexes aren't found in the locations df -" % \
            missing
        print "    this is usually because of a few records that don't join "
        print "    correctly between the locations df and the aggregations tables"

    print "There are %d total movers for this LCM" % len(choosers)
    
    if len(choosers) > vacant_units.sum():
        print "WARNING: Not enough locations for movers"
        print "    reducing locations to size of movers for performance gain"
        choosers = choosers.head(vacant_units.sum())
        
    choices = model.predict(choosers, units, debug=True)
        
    return pd.Series(units.loc[choices.values][model.choice_column].values,
                              index=choices.index)


def simple_transition(tbl, rate, location_fname, set_year_built=False):
    """
    Run a simple growth rate transition model on the table passed in

    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to be transitioned
    rate : float
        Growth rate
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)

    Returns
    -------
    Nothing
    """
    transition = GrowthRateTransition(rate)
    df_base = tbl.to_frame(tbl.local_columns)

    print "%d agents before transition" % len(df_base.index)
    df, added, copied, removed = transition.transition(df_base, None)
    print "%d agents after transition" % len(df.index)

    # Change tracking
    record_change_sets('added', (tbl.name, added))
    record_change_sets('removed', (tbl.name, removed))

    df.loc[added, location_fname] = -1

    if set_year_built:
        df.loc[added, 'year_built'] = orca.get_injectable('year')

    orca.add_table(tbl.name, df)


def record_change_sets(change_type, change_data):
    """
    Record change sets generated by the model steps. 

    Parameters
    ----------
    change_type : str
        Can be "added", "removed", or "updated".
    change_data : tuple
        In the case of additions and removals, change_data is a tuple of form (table_name, index_of_impacted_rows).
        In the case of updates, change_data is a tuple of form (table_name, column_name, Series of updated data).  
            Series of updated data can be a subset of the column if only a subset of rows had updated values.

    Returns
    -------
    None

    """
    if orca.get_injectable('track_changes'):
        year = orca.get_injectable('year')
        model_step = orca.get_injectable('iter_step').step_name
        changes = orca.get_injectable('change_sets')

        if year not in changes.keys():
            changes[year] = {}
            changes[year]['added'] = {}
            changes[year]['removed'] = {}
            changes[year]['updated'] = {}
            orca.add_injectable('change_sets', changes)

        if change_type == 'added':
            record_add(year, model_step, change_data)

        if change_type == 'removed':
            record_removal(year, model_step, change_data)

        if change_type == 'updated':
            record_update(year, model_step, change_data)


def record_add(year, model_step, change_data):
        table_name = change_data[0]
        added_records = change_data[1]

        changes = orca.get_injectable('change_sets')
        changes[year]['added'][(table_name, model_step)] = added_records
        orca.add_injectable('change_sets', changes)


def record_removal(year, model_step, change_data):
        table_name = change_data[0]
        removed_records = change_data[1]

        changes = orca.get_injectable('change_sets')
        changes[year]['removed'][(table_name, model_step)] = removed_records
        orca.add_injectable('change_sets', changes)


def record_update(year, model_step, change_data):
        table_name = change_data[0]
        column_name = change_data[1]
        updated_data = change_data[2]
        
        changes = orca.get_injectable('change_sets')
        changes[year]['updated'][(table_name, column_name, model_step)] = updated_data
        orca.add_injectable('change_sets', changes)


def register_table_from_store(table_name):
    """
    Create orca function for tables from data store.
    """
    @orca.table(table_name, cache=True)
    def func(store):
        return store[table_name]
    return func


def register_config_injectable_from_yaml(injectable_name, yaml_file):
    """
    Create orca function for YAML-based config injectables.
    """
    @orca.injectable(injectable_name, cache=True)
    def func():
        with open(os.path.join(misc.configs_dir(), yaml_file)) as f:
            config = yaml.load(f)
            return config
    return func


def register_simple_transition_model(agents_name, growth_rate):

    @orca.step('simple_%s_transition' % agents_name)
    def simple_transition_model():
        agents_table = orca.get_table(agents_name)
        print 'Running %s transition with %s percent growth rate' % (agents_name, growth_rate*100.0)
        return simple_transition(agents_table, growth_rate, orca.get_injectable('geography_id'))

    return simple_transition_model


def register_choice_model_step(model_name, agents_name):

    @orca.step(model_name)
    def choice_model_simulate(location_choice_models):
        model = location_choice_models[model_name]

        choices = simulate_choice_model(model, choice_function=unit_choices)

        orca.get_table(agents_name).update_col_from_series(model.choice_column, choices)
        
    return choice_model_simulate
