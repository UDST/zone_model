from __future__ import print_function

import os
import copy
import yaml
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

try:
    import joblib
except ModuleNotFoundError as e:
    from sklearn.externals import joblib  # deprecated in sklearn 0.23

import orca
from urbansim.utils import misc
from urbansim.models import dcm
from urbansim.models import util
from urbansim.models import transition
from urbansim.urbanchoice import interaction
from urbansim.models import GrowthRateTransition
from urbansim.models.transition import add_rows
from urbansim.models import MNLDiscreteChoiceModel
from urbansim.models.regression import RegressionModel
from urbansim.models.supplydemand import supply_and_demand


def random_choices(model, choosers, alternatives):
    """
    Simulate choices using random choice, weighted by probability
    but not capacity constrained.
    Parameters
    ----------
    model : SimulationChoiceModel
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
    probabilities = model.calculate_probabilities(choosers, alternatives)
    choices = np.random.choice(
        probabilities.index, size=len(choosers),
        replace=True, p=probabilities.values)
    return pd.Series(choices, index=choosers.index)


def unit_choices(model, choosers, alternatives, enable_supply_correction=None,
                zones_as_alternatives=False, custom_prob_function=None):
    """
    Simulate choices using unit choice.  Alternatives table is expanded
    to be of length alternatives.vacant_variables, then choices are simulated
    from among the universe of vacant units, respecting alternative capacity.
    Parameters
    ----------
    model : SimulationChoiceModel
        Fitted model object.
    choosers : pandas.DataFrame
        DataFrame of choosers.
    alternatives : pandas.DataFrame
        DataFrame of alternatives.
    enable_supply_correction : dict, optional
        Dictionary with parameters for supply-demand price equilibration.
        Must contain keys "price_col" and "submarket_col" which are set to
        the column names in the zones table which contain the column for
        prices and an identifier which segments zones into submarkets
    zones_as_alternatives: boolean, optional
        If zone_id is used as choice column during model estimation but units are
        used as alternatives during model prediction, zones with very large capacities
        can get disproportionately high probabilities. Setting this parameter to True
        corrects for that discrepancy between estimation/prediction contexts by
        aggregating unit-level probabilities by zone_id and then dividing them by
        the number of units in each zone.
    custom_prob_function: func, optional
        Function that can be passed to the unit choices method when the user
        needs to customize the probabilities or choice logic even further than
        what the zones_as_alternatives parameter allows
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    supply_variable, vacant_variable = (model.supply_variable,
                                        model.vacant_variable)
    additional_columns = ['zone_id',supply_variable, vacant_variable]
    if enable_supply_correction is not None \
            and "submarket_col" in enable_supply_correction:
        additional_columns += [enable_supply_correction["submarket_col"]]
    if enable_supply_correction is not None\
            and "price_col" in enable_supply_correction:
        additional_columns += [enable_supply_correction["price_col"]]
    if enable_supply_correction is not None:
        alternatives = orca.get_table('zones')\
            .to_frame(alternatives.columns.tolist() + additional_columns)
    available_units = alternatives[supply_variable]
    vacant_units = alternatives[vacant_variable]
    # must have positive index
    vacant_units = vacant_units[vacant_units.index.values.astype('float') >= 0]

    print("There are {} total available units"
          .format(available_units.sum()),
          "    and {} total choosers"
          .format(len(choosers)),
          "    but there are {} overfull alternatives"
          .format(len(vacant_units[vacant_units < 0])))

    vacant_units = vacant_units[vacant_units > 0]

    indexes = np.repeat(vacant_units.index.values,
                        vacant_units.values.astype('int'))
    isin = pd.Series(indexes).isin(alternatives.index)
    missing = len(isin[isin == False])  # noqa
    indexes = indexes[isin.values]
    units = alternatives.loc[indexes].reset_index()

    print("    for a total of {} temporarily empty units"
          .format(vacant_units.sum()),
          "    in {} alternatives total in the region"
          .format(len(vacant_units)))

    if missing > 0:
        print(
            "WARNING: {} indexes aren't found in the locations df -"
            .format(missing),
            "    this is usually because of a few records that don't join ",
            "    correctly between the locations df and the aggregations",
            "tables")

    print("There are {} total movers for this LCM".format(len(choosers)))

    if enable_supply_correction is not None:
        assert isinstance(enable_supply_correction, dict)
        assert "price_col" in enable_supply_correction
        price_col = enable_supply_correction["price_col"]
        assert "submarket_col" in enable_supply_correction
        submarket_col = enable_supply_correction["submarket_col"]

        if enable_supply_correction.get("warm_start", False) is True:
            raise NotImplementedError()

        multiplier_func = enable_supply_correction.get("multiplier_func", None)
        if multiplier_func is not None:
            multiplier_func = orca.get_injectable(multiplier_func)

        kwargs = enable_supply_correction.get('kwargs', {})
        model.summary_alts_xref = units[submarket_col]
        new_prices, submarkets_ratios = supply_and_demand(
            model,
            choosers,
            units,
            submarket_col,
            price_col,
            base_multiplier=None,
            multiplier_func=multiplier_func,
            **kwargs)


        submarket_table = enable_supply_correction.get("submarket_table", None)
        if submarket_table is not None:
            submarkets_ratios = submarkets_ratios.reindex(
                orca.get_table(submarket_table).index).fillna(1)
            # write final shifters to the submarket_table for use in debugging
            orca.get_table(submarket_table)["price_shifters"] = submarkets_ratios

        print ("Running supply and demand")
        print ("Simulated Prices")
        print (units[price_col].describe())
        print ("Submarket Price Shifters")
        print (submarkets_ratios.describe())
        units[price_col] = units[price_col] * submarkets_ratios.loc[units[submarket_col]].values
        prices = units.groupby('zone_id')[price_col].min().reset_index()
        alternatives = alternatives.rename(columns={price_col: price_col+"_hedonic"})
        alternatives = alternatives.merge(prices, on='zone_id', how='left').set_index('zone_id')
        alternatives.loc[alternatives[price_col].isnull(), price_col] = alternatives[price_col+"_hedonic"]
        orca.add_column('zones', price_col+"_hedonic", alternatives[price_col+"_hedonic"])
        orca.add_column('zones', price_col , alternatives[price_col])
        print ("Adjusted Prices")
        print (alternatives[price_col].describe())

    if len(choosers) > vacant_units.sum():
        print("WARNING: Not enough locations for movers",
              "reducing locations to size of movers for performance gain")
        print ("{} choosers will not be placed".format(len(choosers) - vacant_units.sum()))
        choosers = choosers.head(int(vacant_units.sum()))
    if custom_prob_function:
        choices = custom_prob_function(model, choosers, units, zones_as_alternatives)
    elif zones_as_alternatives==True:
        choices = choose_with_zones_as_alternatives(model, choosers, units)
    else:
        choices = model.predict(choosers, units, debug=True)

    def identify_duplicate_choices(choices):
        choice_counts = choices.value_counts()
        return choice_counts[choice_counts > 1].index.values

    if model.choice_mode == 'individual':
        print('Choice mode is individual, so utilizing lottery choices.')

        chosen_multiple_times = identify_duplicate_choices(choices)

        while len(chosen_multiple_times) > 0:
            duplicate_choices = choices[choices.isin(chosen_multiple_times)]

            # Identify the choosers who keep their choice, and those who must
            # choose again.
            keep_choice = duplicate_choices.drop_duplicates()
            rechoose = duplicate_choices[~duplicate_choices.index.isin(
                                                           keep_choice.index)]

            # Subset choices, units, and choosers to account for occupied
            # units and choosers who need to choose again.
            choices = choices.drop(rechoose.index)
            units_remaining = units.drop(choices.values)
            choosers = choosers.drop(choices.index)

            # Agents choose again.
            next_choices = model.predict(choosers, units_remaining)
            choices = pd.concat([choices, next_choices])
            chosen_multiple_times = identify_duplicate_choices(choices)

    if enable_supply_correction is not None:
        new_prices = alternatives[price_col]
        if "clip_final_price_low" in enable_supply_correction:
            new_prices = new_prices.clip(lower=enable_supply_correction[
                "clip_final_price_low"])
        if "clipal_final_price_high" in enable_supply_correction:
            new_prices = new_prices.clip(upper=enable_supply_correction[
                "clip_final_price_high"])
        orca.add_column('zones', price_col, new_prices)

    return pd.Series(units.loc[choices.values][model.choice_column].values,
                     index=choices.index)

def choose_with_zones_as_alternatives(model, choosers, units):
    model.predict(choosers, units, debug=True)
    orig_probas = model.sim_pdf.reset_index()
    orig_probas = orig_probas.rename(columns={0: 'orig_prob'})
    choosers, units = model.apply_predict_filters(choosers, units)
    units.index.names = ['alternative_id']
    orig_probas = orig_probas.merge(
        units[['zone_id']].reset_index(), on='alternative_id', how='left')
    alts_per_zone = orig_probas.groupby('zone_id')['alternative_id'].count().reset_index()
    alts_per_zone = alts_per_zone.rename(columns={'alternative_id': 'alts_zone'})
    orig_probas = orig_probas.merge(alts_per_zone, on='zone_id', how='left')
    units_ = units.groupby('zone_id').min().reset_index()
    choosers_ = choosers.iloc[:1, :]
    model.predict(choosers_, units_.set_index('zone_id'), debug=True)
    agg_probas = model.sim_pdf.reset_index()
    agg_probas = agg_probas.rename(
        columns={'alternative_id': 'zone_id', 0: 'agg_prob'})
    probas = orig_probas.merge(agg_probas, on='zone_id', how='left')
    probas['probability'] = probas['agg_prob'] / probas['alts_zone']
    choices = dcm.unit_choice(choosers.index.values,
                              probas.alternative_id.values,
                              probas.probability.values)
    return choices

def lottery_choices_agent_units(model, choosers, alternatives, max_iter=15):
    """
    Simulate choices using lottery choices.  Alternatives are selected
    iteratively with agent_units respecting capacities until all agents
    are placed, capacity is zero, or max iterations is reached.
    Parameters
    ----------
    model : SimulationChoiceModel
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

    supply_variable, vacant_variable, agent_units = (model.supply_variable,
                                                     model.vacant_variable,
                                                     model.agent_units)

    available_units = alternatives[supply_variable]
    vacant_units = alternatives[vacant_variable]

    print("There are {} total available units"
          .format(available_units.sum()),
          "and {} total choosers"
          .format(len(choosers)),
          "but there are {} overfull alternatives"
          .format(len(vacant_units[vacant_units < 0])))

    choices = model.predict(choosers, alternatives)
    choosers['new_choice_id'] = choices

    def vacancy_check(vacant_units, choosers, agent_units):
        unit_check = (vacant_units -
                      choosers.groupby('new_choice_id')[agent_units].sum())
        over = unit_check[unit_check < 0]
        return unit_check, over

    unit_check, over = vacancy_check(vacant_units, choosers, agent_units)
    iteration = 2

    while (len(over) > 0) & (iteration <= max_iter):
        iteration += 1
        choose_again = np.array([])
        for ialt in over.index:
            idx = choosers.index[choosers.new_choice_id == ialt]
            units = choosers[agent_units][choosers.new_choice_id == ialt]
            cap = alternatives[vacant_variable][alternatives.index == ialt]
            permutate = np.random.permutation(idx.size)
            csum = units[idx[permutate]].cumsum()
            draw = idx[permutate[np.where(csum > cap[ialt])]]
            choose_again = np.concatenate((choose_again, draw))
        chosen = choosers.loc[~choosers.index.isin(choose_again)]
        still_choosing = choosers.loc[choosers.index.isin(choose_again)]
        chosen_sum = chosen.groupby('new_choice_id')[agent_units].sum()
        unit_check = pd.DataFrame(data={'vac': vacant_units,
                                        'chosen_sum': chosen_sum})
        unit_check['new_vacancy'] = (unit_check.vac -
                                     unit_check.chosen_sum).fillna(
                                     unit_check.vac)

        full = unit_check.index[unit_check.new_vacancy <= 1]
        alternatives = alternatives[~alternatives.index.isin(full)]
        choices = model.predict(still_choosing, alternatives)
        choosers.loc[choosers.index.isin(choices.index),
                     'new_choice_id'] = choices
        unit_check, over = vacancy_check(vacant_units, choosers, agent_units)
    if len(choosers) > 0:
        print("Placed {} {} with {} {} in {} iterations"
              .format(len(chosen), model.choosers,
                      chosen[agent_units].sum(), agent_units,
                      iteration-1))
        print("{} unplaced {} remain with {} {}"
              .format(len(over), model.choosers,
                      int(choosers.loc[choosers.index.isin(over.index),
                                       [agent_units]].sum()), agent_units))

    choosers.loc[choosers.index.isin(over.index), 'new_choice_id'] = -1
    return choosers.new_choice_id


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

    print("{} agents before transition".format(len(df_base.index)))
    df, added, copied, removed = transition.transition(df_base, None)
    print("{} agents after transition".format(len(df.index)))

    # Change tracking
    record_change_sets('added', (tbl.name, added))
    record_change_sets('removed', (tbl.name, removed))

    df.loc[added, location_fname] = -1

    if set_year_built:
        df.loc[added, 'year_built'] = orca.get_injectable('year')

    orca.add_table(tbl.name, df)


def full_transition(agents, agent_controls, totals_column, year,
                    location_fname, linked_tables=None,
                    accounting_column=None, set_year_built=False):
    """
    Run a transition model based on control totals specified in the usual
    UrbanSim way

    Parameters
    ----------
    agents : DataFrameWrapper
        Table to be transitioned
    agent_controls : DataFrameWrapper
        Table of control totals
    totals_column : str
        String indicating the agent_controls column to use for totals.
    year : int
        The year, which will index into the controls
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    linked_tables : dict, optional
        Sets the tables linked to new or removed agents to be updated with
        dict of {'table_name':(DataFrameWrapper, 'link_id')}
    accounting_column : str, optional
        Name of column with accounting totals/quantities to apply toward the
        control. If not provided then row counts will be used for accounting.
    set_year_built: boolean
        Indicates whether to update 'year_built' columns with current
        simulation year

    Returns
    -------
    Nothing
    """
    ct = agent_controls.to_frame()
    agnt = agents.to_frame()
    print("Total agents before transition: {}".format(len(agnt)))
    tran = transition.TabularTotalsTransition(ct, totals_column,
                                              accounting_column)
    updated, added, copied, removed = tran.transition(agnt, year)
    updated.loc[added, location_fname] = -1
    if set_year_built:
        updated.loc[added, 'year_built'] = year

    updated_links = {}
    if linked_tables:
        for table_name, (table, col) in linked_tables.items():
            print('updating linked table {}'.format(table_name))
            updated_links[table_name] = \
                update_linked_table(table, col, added, copied, removed)
            orca.add_table(table_name, updated_links[table_name])

    print("Total agents after transition: {}".format(len(updated)))
    orca.add_table(agents.name, updated[agents.local_columns])
    return updated, added, copied, removed


def update_linked_table(tbl, col_name, added, copied, removed):
    """
    Copy and update rows in a table that has a column referencing another
    table that has had rows added via copying.

    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to update with new or removed rows.
    col_name : str
        Name of column in `table` that corresponds to the index values
        in `copied` and `removed`.
    added : pandas.Index
        Indexes of rows that are new in the linked table.
    copied : pandas.Index
        Indexes of rows that were copied to make new rows in linked table.
    removed : pandas.Index
        Indexes of rows that were removed from the linked table.

    Returns
    -------
    updated : pandas.DataFrame

    """
    print('start: update linked table after transition')

    # handle removals
    table = tbl.local
    table = table.loc[~table[col_name].isin(set(removed))]
    removed = table.loc[table[col_name].isin(set(removed))]
    if (added is None or len(added) == 0):
        return table

    # map new IDs to the IDs from which they were copied
    id_map = pd.concat([pd.Series(copied, name=col_name),
                        pd.Series(added, name='temp_id')], axis=1)

    # join to linked table and assign new id
    new_rows = id_map.merge(table, on=col_name)
    new_rows.drop(col_name, axis=1, inplace=True)
    new_rows.rename(columns={'temp_id': col_name}, inplace=True)

    # index the new rows
    starting_index = table.index.values.max() + 1
    new_rows.index = np.arange(starting_index,
                               starting_index + len(new_rows), dtype=np.int)

    if orca.get_injectable('track_changes'):
        add_data = (tbl.name, added)
        record_change_sets("added", add_data)
        remove_data = (tbl.name, removed.index)
        record_change_sets("removed", remove_data)

    print('finish: update linked table after transition')
    return pd.concat([table, new_rows])


def vacancy_rate_targets(buildings, target_vacancies):
    """
    Calculate new unit targets for vacancy rate transition model.

    Parameters
    ----------
    buildings : DataFrameWrapper
        Buildings table dataframe wrapper
    target_vacancies : DataFrameWrapper
        Sarget vacancy table wrapper with columns for 'building_type_id',
        'target_vacancy' proportion, and boolean 'residential'.

    Returns
    -------
    bsum : DataFrame
        Summary table of new units required per building type to meet
        vacancy rate expectations.

    """
    b = buildings.to_frame(['building_type_id','residential_units',
                            'vacant_residential_units','job_spaces',
                            'vacant_job_spaces'])
    targets = target_vacancies.to_frame()
    bsum = b.groupby('building_type_id').sum()
    bsum['target_vacancy'] = targets.target_vacancy
    bsum['residential'] = targets.residential
    bsum['res_vacancy_rate'] = bsum.vacant_residential_units / bsum.residential_units
    bsum['nonres_vacancy_rate'] = bsum.vacant_job_spaces / bsum.job_spaces
    bsum['res_target'] = np.round(bsum.residential_units * np.clip((
        (bsum.target_vacancy - bsum.res_vacancy_rate) + 1), 1, np.inf))
    bsum['nonres_target'] = np.round(bsum.job_spaces * np.clip((
        (bsum.target_vacancy - bsum.nonres_vacancy_rate) + 1), 1, np.inf))
    bsum['new_res'] = bsum.res_target - bsum.residential_units
    bsum['new_nonres'] = bsum.nonres_target - bsum.job_spaces
    return bsum[['res_vacancy_rate','nonres_vacancy_rate','target_vacancy',
                 'new_nonres','new_res','residential']]


def vacancy_rate_transition(buildings, target_vacancies,
                            development_event_history,
                            developments_table_name):
    """
    Sample new development project from a development history table based
    on target vacancy rates and add them to a developments table.

    Parameters
    ----------
    buildings : DataFrameWrapper
        Buildings table dataframe wrapper
    target_vacancies : DataFrameWrapper
        Target vacancy table wrapper with columns for 'building_type_id',
        'target_vacancy' proportion, and boolean 'residential'.
    development_event_history : DataFrameWrapper
        Historical development projects to sample from.
    developments_table_name : str
        The name of the developments table to add sampled projects to.

    Returns
    -------
    None

    """
    targets = vacancy_rate_targets(buildings, target_vacancies)
    dev_events = development_event_history.to_frame()
    for i, r in targets.iterrows():
        units = 'residential_units' if r.residential==1 else 'job_spaces'
        vac = 'res_vacancy_rate' if r.residential==1 else 'nonres_vacancy_rate'
        target = r['new_res'] if r.residential==1 else r['new_nonres']
        dev = dev_events.loc[dev_events.building_type_id==i]
        print("current vacancy for building type {}: {}% " \
              "target vacancy for building type {}: {}%".format(i, np.round(r[vac] * 100,3),
                                                               i, np.round(r.target_vacancy * 100,3)))
        if r.target_vacancy < r[vac]:
            print("no new {} required".format(units))
        else:
            updated, added, copied = add_rows(dev, target, accounting_column=units)
            print("adding {} developments of type {} with {} {}".format(
                   len(added), i, updated.loc[added, units].sum(), units))
            updated = updated[['building_type_id','residential_units','non_residential_sqft']]
            dev = orca.get_table(developments_table_name).to_frame()
            dev = pd.concat([dev, updated.loc[added]])
            orca.add_table(developments_table_name, dev)


def record_change_sets(change_type, change_data):
    """
    Record change sets generated by the model steps.

    Parameters
    ----------
    change_type : str
        Can be "added", "removed", or "updated".
    change_data : tuple
        In the case of additions and removals, change_data is a tuple of form
        (table_name, index_of_impacted_rows).
        In the case of updates, change_data is a tuple of form
        (table_name, column_name, Series of updated data).
        Series of updated data can be a subset of the column
        if only a subset of rows had updated values.

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
    changes[year]['updated'][
        (table_name, column_name, model_step)] = updated_data
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
            config = yaml.safe_load(f)
            return config
    return func


def register_simple_transition_model(agents_name, growth_rate):

    @orca.step('simple_%s_transition' % agents_name)
    def simple_transition_model():
        agents_table = orca.get_table(agents_name)
        print('Running {} transition with {:.2f} percent growth rate'
              .format(agents_name, growth_rate * 100.0))
        return simple_transition(agents_table, growth_rate,
                                 orca.get_injectable('geography_id'))

    return simple_transition_model


def register_choice_model_step(model_name, agents_name, choice_function,
                               enable_supply_correction=None,
                               zones_as_alternatives=False):

    @orca.step(model_name)
    def choice_model_simulate(location_choice_models):
        model = location_choice_models[model_name]

        choices = model.simulate(choice_function=choice_function,
                                 enable_supply_correction=enable_supply_correction,
                                 zones_as_alternatives=zones_as_alternatives)

        # Test if the simulation was performed
        if not (choices is None):
            print('There are {} more unplaced agents.'.format(choices.isnull().sum()))
            orca.get_table(agents_name).update_col_from_series(
                    model.choice_column, choices, cast=True)
    return choice_model_simulate


def register_regression_model_step(model_name):

    @orca.step(model_name)
    def regression_model_simulate(regression_models):
        model = regression_models[model_name]
        dataset_name = model.observations_name
        df = orca.get_table(dataset_name).to_frame(model.columns_used())

        predictions = model.predict(df)

        print('Summary of model predictions for {}: {}'
              .format(model_name, predictions.describe()))

        orca.get_table(dataset_name).update_col_from_series(
                                      model.dep_var, predictions, cast=True)

    return regression_model_simulate


class SimulationChoiceModel(MNLDiscreteChoiceModel):
    """
    A discrete choice model with parameters needed for simulation.
    Initialize with MNLDiscreteChoiceModel's init parameters or with from_yaml,
    then add simulation parameters with set_simulation_params().
    """
    def set_simulation_params(self, name, supply_variable, vacant_variable,
                              choosers, alternatives, choice_column=None,
                              summary_alts_xref=None, merge_tables=None,
                              agent_units=None, calibrated=False,
                              min_chooser_cols=False, equilibration=False):
        """
        Add simulation parameters as additional attributes.
        Parameters
        ----------
        name : str
            Name of the model.
        supply_variable : str
            The name of the column in the alternatives table indicating number
            of available spaces, vacant or not, that can be occupied by
            choosers.
        vacant_variable : str
            The name of the column in the alternatives table indicating number
            of vacant spaces that can be occupied by choosers.
        choosers : str
            Name of the choosers table.
        alternatives : str
            Name of the alternatives table.
        summary_alts_xref : dict or pd.Series, optional
            Mapping of alternative index to summary alternative id.  For use
            in evaluating a model with many alternatives.
        merge_tables : list of str, optional
            List of additional tables to be broadcast onto the alternatives
            table.
        agent_units : str, optional
            Name of the column in the choosers table that designates how
            much supply is occupied by each chooser.
        calibrated : bool, optional
            Indicates whether or not model includes calibrated coefficients.
        min_chooser_cols : bool, optional
            Indicates whether to calculate minimum choose variables.  Note that
            this precludes some chooser * alternative interaction variables.
        equilibration : bool, optional
            Indicates whether supply-demand price equilibration should be
            applied to the model.
        Returns
        -------
        None
        """
        self.name = name
        self.supply_variable = supply_variable
        self.vacant_variable = vacant_variable
        self.choosers = choosers
        self.alternatives = alternatives
        self.summary_alts_xref = summary_alts_xref
        self.merge_tables = merge_tables
        self.agent_units = agent_units
        self.choice_column = choice_column if choice_column is not None \
            else self.choice_column
        self.calibrated = calibrated
        self.min_chooser_cols = min_chooser_cols
        self.equilibration = equilibration


    def set_calibration_variables(self, calib_variables):
        """
        Update model expression and fit parameters with calibrated variables.
        Parameters
        ----------
        calib_variables : dict
            Mapping between variable name and coefficient.
        Returns
        -------
        None
        """
        self.calibrated = True
        for calib_var in calib_variables.keys():
            calib_var_name = calib_var.replace('_x_', ':')
            if calib_var_name not in self.model_expression:
                if type(self.model_expression) == set:
                    self.model_expression.add(calib_var_name)
                else:
                    self.model_expression.append(calib_var_name)
                print('Adding calib coeffs: %s' % calib_var_name)
                coeff = float(calib_variables[calib_var])

                to_add = {'Coefficient':coeff, 'Std. Error':0.0,
                                                  'T-Score':0.0}
                to_add = pd.Series(to_add, name=calib_var_name)
                self.fit_parameters = self.fit_parameters.append(to_add)


    def simulate(self, choice_function=None, save_probabilities=False,
                 **kwargs):
        """
        Computing choices, with arbitrary function for handling simulation
        strategy.
        Parameters
        ----------
        choice_function : function
            Function defining how to simulate choices based on fitted model.
            Function must accept the following 3 arguments:  model object,
            choosers DataFrame, and alternatives DataFrame. Additional optional
            keyword args can be utilized by function if needed (kwargs).
        save_probabilities : bool
            If true, will save the calculated probabilities underlying the
            simulation as an orca injectable with name
            'probabilities_modelname_itervar'.
        Returns
        -------
        choices : pandas.Series
            Mapping of chooser ID to alternative ID. Some choosers
            will map to a nan value when there are not enough alternatives
            for all the choosers.
        """
        # If there are no choosers, return None without simulating
        chooser_cols = self.choosers_columns_used()
        chooser_cols.append(self.choice_column)
        choosers = orca.get_table(self.choosers).to_frame(chooser_cols)
        choosers = choosers.query(self.choosers_predict_filters)
        choice_col_check = choosers[self.choice_column]
        if (choice_col_check == -1).sum() == 0:
            print('There are no choosers for model {}.'.format(self.name))
            return None

        choosers, alternatives = self.calculate_model_variables()

        choosers, alternatives = self.apply_predict_filters(
                                 choosers, alternatives)

        # By convention, choosers are denoted by a -1 value
        # in the choice column
        choosers = choosers[choosers[self.choice_column] == -1]
        print("{} agents are making a choice.".format(len(choosers)))

        if choice_function:
            choices = choice_function(self, choosers, alternatives, **kwargs)
        else:
            choices = self.predict(choosers, alternatives, debug=True)

        if save_probabilities:
            if not self.sim_pdf:
                probabilities = self.calculate_probabilities(choosers,
                                                             alternatives)
            else:
                probabilities = self.sim_pdf.reset_index().set_index(
                    'alternative_id')[0]
            orca.add_injectable('probabilities_{}_{}'.format(
                self.name, orca.get_injectable('iter_var')),
                probabilities)

        return choices

    def fit_model(self):
        """
        Estimate model based on existing parameters
        Returns
        -------
        None
        """
        choosers, alternatives = self.calculate_model_variables()
        self.fit(choosers, alternatives, choosers[self.choice_column])
        return self.log_likelihoods, self.fit_parameters

    def calculate_probabilities(self, choosers=None, alternatives=None):
        """
        Calculate model probabilities.
        Parameters
        ----------
        choosers : pandas.DataFrame
            DataFrame of choosers.
        alternatives : pandas.DataFrame
            DataFrame of alternatives.
        Returns
        -------
        probabilities : pandas.Series
            Mapping of alternative ID to probabilities.
        """
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        probabilities = self.probabilities(choosers, alternatives)
        probabilities = probabilities.reset_index().set_index(
            'alternative_id')[0]  # remove chooser_id col from idx
        return probabilities

    def calculate_model_variables(self):
        """
        Calculate variables needed to simulate the model, and returns
        DataFrames of simulation-ready tables with needed variables.
        Returns
        -------
        choosers : pandas.DataFrame
            DataFrame of choosers.
        alternatives : pandas.DataFrame
            DataFrame of alternatives.
        """
        columns_used = self.columns_used() + [self.choice_column]

        if self.min_chooser_cols:
            chooser_cols = self.choosers_columns_used()
            chooser_cols.append(self.choice_column)
            choosers = orca.get_table(self.choosers).to_frame(chooser_cols)
        else:
            choosers = orca.get_table(self.choosers).to_frame(columns_used)

        supply_column_names = [col for col in
                               [self.supply_variable, self.vacant_variable]
                               if col is not None]

        columns_used.extend(supply_column_names)

        if self.merge_tables:
            mt = copy.deepcopy(self.merge_tables)
            mt.append(self.alternatives)
            all_cols = []
            for table in mt:
                all_cols.extend(orca.get_table(table).columns)
            all_cols = [col for col in all_cols if col in columns_used]
            alternatives = orca.merge_tables(target=self.alternatives,
                                             tables=mt, columns=all_cols)
        else:
            alternatives = orca.get_table(self.alternatives).to_frame(
                columns_used + supply_column_names)
        return choosers, alternatives

    def score(self, scoring_function=accuracy_score, choosers=None,
              alternatives=None, aggregate=False, apply_filter=True,
              choice_function=random_choices):
        """
        Calculate score for model.  Defaults to accuracy score, but other
        scoring functions can be provided.  Computed on all choosers/
        alternatives by default, but can also be computed on user-supplied
        test datasets.  If model has a summary_alts_xref, then score
        calculated after mapping to summary ids.
        Parameters
        ----------
        scoring_function : function, default sklearn.metrics.accuracy_score
            Function defining how to score model predictions. Function must
            accept the following 2 arguments:  pd.Series of observed choices,
            pd.Series of predicted choices.
        choosers : pandas.DataFrame, optional
            DataFrame of choosers.
        alternatives : pandas.DataFrame, optional
            DataFrame of alternatives.
        aggregate : bool
            Whether to calculate score based on total count of choosers that
            made each choice, rather than based on disaggregate choices.
        apply_filter : bool
            Whether to apply the model's choosers_predict_filters prior to
            calculating score.  If supplying own test dataset, and do not want
            it further manipulated, then set to False.
        choice_function : function, option
            Function defining how to simulate choices.
        Returns
        -------
        score : float
            The model's score (accuracy score by default).
        """
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if apply_filter:
            if self.choosers_predict_filters:
                choosers = choosers.query(self.choosers_predict_filters)
            if self.choosers_fit_filters:
                choosers = choosers.query(self.choosers_fit_filters)

        observed_choices = choosers[self.choice_column]
        predicted_choices = choice_function(self, choosers, alternatives)

        if self.summary_alts_xref is not None:
            observed_choices = observed_choices.map(self.summary_alts_xref)
            predicted_choices = predicted_choices.map(self.summary_alts_xref)

        if aggregate:
            observed_choices = observed_choices.value_counts()
            predicted_choices = predicted_choices.value_counts()

            combined_index = list(set(list(predicted_choices.index) +
                                      list(observed_choices.index)))
            predicted_choices = predicted_choices.reindex(combined_index).fillna(0)
            observed_choices = observed_choices.reindex(combined_index).fillna(0)

        return scoring_function(observed_choices, predicted_choices)

    def summed_probabilities_for_score(self, choosers=None, alternatives=None):
        """
        Sum probabilities to the summary geography level.
        """
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        choosers['summary_id'] = choosers[self.choice_column]
        choosers.summary_id = choosers.summary_id.map(self.summary_alts_xref)
        probs = self.calculate_probabilities(choosers, alternatives)
        probs = probs.reset_index().rename(columns={0: 'proba'})
        probs['summary_id'] = probs.alternative_id.map(self.summary_alts_xref)
        return probs.groupby('summary_id').proba.sum()

    def summed_probabilities(self, choosers=None, alternatives=None):
        summed_probas = dcm.MNLDiscreteChoiceModel.summed_probabilities(self, choosers, alternatives)
        return summed_probas


    def observed_distribution(self, choosers=None):
        """
        Calculate observed distribution across alternatives at the summary
        geography level.
        """
        if choosers is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        if 'summary_id' not in choosers.columns:
            summ_id = choosers[self.choice_column].map(self.summary_alts_xref)
            choosers['summary_id'] = summ_id

        observed_distrib = choosers.groupby('summary_id').size()
        return observed_distrib / observed_distrib.sum()

    def summed_probability_score(self, scoring_function=r2_score,
                                 choosers=None, alternatives=None,
                                 validation_data=None):
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        if self.choosers_fit_filters:
            choosers = choosers.query(self.choosers_fit_filters)

        if self.choosers_predict_filters:
            choosers = choosers.query(self.choosers_predict_filters)

        summed_probas = self.summed_probabilities_for_score(choosers, alternatives)

        if validation_data is None:
            validation_data = self.observed_distribution(choosers)

        combined_index = list(set(list(summed_probas.index) +
                                  list(validation_data.index)))
        summed_probas = summed_probas.reindex(combined_index).fillna(0)
        validation_data = validation_data.reindex(combined_index).fillna(0)

        print(summed_probas.corr(validation_data))
        score = scoring_function(validation_data, summed_probas)
        print(score)

        residuals = summed_probas - validation_data
        return score, residuals


class SimpleEnsemble(SimulationChoiceModel):
    """
    An ensemble of choice models that allows for simulation with weighted
    average of component choice models.

    Parameters
    ----------
    model_names : list of str
        Name of the models that will comprise the ensemble.
    model_weights : list of float
        The weight to apply to each of the component models.  Should be the
        same length as model_names and should sum to 1.
    """
    def __init__(self, models, model_weights):
        self.models = models
        self.model_weights = model_weights
        self.choice_column = self.models[0].choice_column

    def calculate_probabilities(self, choosers, alternatives):
        """
        Take the weighted average of component model probabilities.
        """
        probabilities = np.asarray(
            [model.calculate_probabilities(choosers, alternatives)
             for model in self.models])
        avg = np.average(probabilities, axis=0, weights=self.model_weights)

        return pd.Series(data=avg, index=alternatives.index)

    def calculate_model_variables(self):
        """
        Calculate all variables needed across component models in ensemble.
        """
        first_model = self.models[0]

        variables = [variable for model in self.models
                     for variable in model.columns_used()]
        columns_used = variables + [self.choice_column]
        choosers = orca.get_table(first_model.choosers).to_frame(columns_used)

        supply_column_names = [first_model.supply_variable,
                               first_model.vacant_variable]
        alternatives = orca.get_table(first_model.alternatives).to_frame(
            columns_used + supply_column_names)
        return choosers, alternatives


def create_lcm_training_data(choosers, alternatives, alts_sample_size,
                             current_choice, model_expression=None):
    """Create training dataset for scikit-learn-based location choice models"""

    current_choice = choosers[current_choice]

    _, merged, chosen = interaction.mnl_interaction_dataset(
                choosers, alternatives, alts_sample_size, current_choice)
    if model_expression is not None:
        model_expression = list(alternatives.columns.values)
        str_model_expression = util.str_model_expression(model_expression,
                                                         add_constant=False)
        model_design = dmatrix(str_model_expression, data=merged,
                               return_type='dataframe')
        X = model_design.as_matrix()
        alt_index = model_design.index
    else:
        X = merged[merged.columns[:-2]].as_matrix()
        alt_index = merged.index

    y = chosen.ravel()

    return X, y, alt_index


class SklearnLocationModel:
    """Model location choice with scikit-learn models"""

    def __init__(self, clf_class, choice_column, summary_alts_xref=None,
                 exp_vars=None, scaler=None, lcm=None, feature_space=None,
                 **kwargs):
        self.clf_class = clf_class
        self.clf = clf_class(**kwargs)
        self.exp_vars = exp_vars
        self.choice_column = choice_column
        self.summary_alts_xref = summary_alts_xref
        self.scaler = scaler
        self.lcm = lcm
        if lcm:
            self.choice_mode = 'individual'
            self.supply_variable = self.lcm.supply_variable
            self.vacant_variable = self.lcm.vacant_variable
            self.name = self.lcm.name
            self.choosers = self.lcm.choosers
        self.feature_space = feature_space
        self.numeric_subsetted = False

    def calculate_model_variables(self, sim=False):
        if sim:
            supply_column_names = [col for col in
                                   [self.supply_variable,
                                    self.vacant_variable]
                                   if col is not None]

        if self.exp_vars:
            columns = self.exp_vars
            if sim:
                columns = columns + supply_column_names
            alts = orca.get_table(self.lcm.alternatives).to_frame(columns)

        else:
            alts = orca.get_table(self.lcm.alternatives)
            alts = alts.to_frame(self.feature_space)
            if not self.numeric_subsetted:
                numerics = ['int16', 'int32', 'int64', 'float16',
                            'float32', 'float64']
                alts = alts.select_dtypes(include=numerics)
                self.feature_space = list(alts.columns.values)
                self.numeric_subsetted = True

        columns_used = self.lcm.columns_used() + [self.lcm.choice_column]
        choosers = orca.get_table(self.lcm.choosers).to_frame(columns_used)
        if not sim:
            if self.lcm.choosers_fit_filters:
                choosers = choosers.query(self.lcm.choosers_fit_filters)
            if self.lcm.choosers_predict_filters:
                choosers = choosers.query(self.lcm.choosers_predict_filters)

        return choosers, alts

    def fit(self, choosers, alternatives, alts_sample_size, current_choice,
            scaler=None):
        X, y, location_id = create_lcm_training_data(choosers,
                                                     alternatives,
                                                     alts_sample_size,
                                                     current_choice,
                                                     self.exp_vars)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                            random_state=0)

        if self.scaler:
            scaler = self.scaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            self.scaler = scaler

        self.clf.fit(X_train, y_train)
        print("Training set score: {:f}".format(self.clf.score(X_train,
                                                               y_train)))
        print("Test set score: {:f}".format(self.clf.score(X_test, y_test)))
        return self.clf

    def calculate_probabilities(self, choosers, alternatives):
        alts_index = alternatives.index
        if self.exp_vars is not None:
            alternatives = alternatives[self.exp_vars]

        alternatives = alternatives.fillna(0).as_matrix()
        alternatives[alternatives == np.inf] = 0

        if self.scaler:
            alternatives = self.scaler.transform(alternatives)

        try:
            loc_score = pd.DataFrame(self.clf.predict_proba(alternatives))[1]
            norm_probas = pd.Series((loc_score / loc_score.sum()).values,
                                    index=alts_index)
        except Exception:
            loc_score = pd.Series(self.clf.decision_function(alternatives))
            if loc_score.min() < 0:
                loc_score = loc_score + abs(loc_score.min())
            norm_probas = pd.Series((loc_score / loc_score.sum()).values,
                                    index=alts_index)

        return norm_probas

    def simulate(self, choice_function=random_choices, choosers=None,
                 alternatives=None):
        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables(sim=True)

        choosers, alternatives = self.lcm.apply_predict_filters(
                                         choosers, alternatives)

        choosers = choosers[choosers[self.choice_column] == -1]
        print("{} agents are making a choice.".format(len(choosers)))

        choices = choice_function(self, choosers, alternatives)
        return choices

    def predict(self, choosers, alternatives, debug=True):
        if len(choosers) == 0:
            return pd.Series()

        if len(alternatives) == 0:
            return pd.Series(index=choosers.index)

        probabilities = self.calculate_probabilities(
                              choosers[[self.choice_column]],
                              alternatives[self.exp_vars])

        if debug:
            self.sim_pdf = probabilities

        choices = dcm.unit_choice(
             choosers.index.values,
             probabilities.index.values,
             probabilities.values)
        return choices

    def score(self, scoring_function=accuracy_score, choosers=None,
              alternatives=None, aggregate=False, apply_filter=True,
              choice_function=random_choices):

        if choosers is None or alternatives is None:
            choosers, alternatives = self.calculate_model_variables()

        observed_choices = choosers[self.choice_column]
        predicted_choices = choice_function(self, choosers, alternatives)

        if self.summary_alts_xref is not None:
            observed_choices = observed_choices.map(self.summary_alts_xref)
            predicted_choices = predicted_choices.map(self.summary_alts_xref)

        if aggregate:
            observed_choices = observed_choices.value_counts()
            predicted_choices = predicted_choices.value_counts()

        return scoring_function(observed_choices.sort_index(), predicted_choices.sort_index())

    def summed_probability_score(self):
        choosers, alts = self.calculate_model_variables()
        if self.exp_vars:
            alts = alts[self.exp_vars]
        else:
            alts = alts[self.feature_space]
        probas = self.calculate_probabilities(choosers, alts)
        probas = probas.reset_index().rename(columns={0: 'proba'})
        summ_id = probas[self.choice_column].map(self.summary_alts_xref)
        probas['summary_id'] = summ_id
        summed_probas = probas.groupby('summary_id').proba.sum()

        validation_data = self.lcm.observed_distribution()

        combined_index = list(set(list(summed_probas.index) +
                                  list(validation_data.index)))
        summed_probas = summed_probas.reindex(combined_index).fillna(0)
        validation_data = validation_data.reindex(combined_index).fillna(0)

        print(summed_probas.corr(validation_data))
        score = r2_score(validation_data, summed_probas)
        print(score)

        residuals = summed_probas - validation_data
        return score, residuals

    def calculate_feature_importance(self):
        features_by_importance = []
        for feature in zip(self.feature_space, self.clf.feature_importances_):
            features_by_importance.append(feature)

        return pd.DataFrame(features_by_importance, columns=['variable', 'fi'])

    def select_n_most_important_features(self, n=10):
        feature_importance = self.calculate_feature_importance()
        top_n = feature_importance.sort_values('fi', ascending=False).head(n)
        return list(top_n.variable.values)

    def to_joblib_pkl(self, model_name):
        joblib.dump(self, '{}.pkl'.format(model_name))

    def from_joblib_pkl(self, model_name):
        pass
        # self.clf = joblib.load('{}.pkl'.format(model_name))


class RegressionProbabilityModel:
    """Model agent location choice with share-regression models
       determining the probabilities."""

    def __init__(self, config_path, lcm=None):
        if config_path.endswith('.yaml'):
            self.rm = RegressionModel.from_yaml(str_or_buffer=yaml_config_path)
            self.model_type = 'urbansim'
            self.exp_vars = self.rm.columns_used()
        else:
            self.rm = joblib.load(config_path)
            self.model_type = 'sklearn'
            self.exp_vars = self.rm.exp_vars
        self.lcm = lcm
        if lcm:
            self.choice_mode = 'individual'
            self.supply_variable = self.lcm.supply_variable
            self.vacant_variable = self.lcm.vacant_variable
            self.name = self.lcm.name
            self.choosers = self.lcm.choosers
            self.choice_column = self.lcm.choice_column

    def calculate_model_variables(self):
        supply_column_names = [col for col in
                               [self.supply_variable,
                                self.vacant_variable]
                               if col is not None]
        columns_used = self.exp_vars + supply_column_names
        alts = orca.get_table(self.lcm.alternatives).to_frame(columns_used)

        columns_used = self.lcm.columns_used() + [self.lcm.choice_column]
        choosers = orca.get_table(self.lcm.choosers).to_frame(columns_used)

        return choosers, alts

    def calculate_probabilities(self, chooser, alternatives):
        alternatives = alternatives[self.exp_vars]
        predicted_probas = self.rm.predict(alternatives)
        if self.model_type == 'sklearn':
            predicted_probas = pd.Series(predicted_probas,
                                         index=alternatives.index)
        min_proba = predicted_probas.min()
        if min_proba < 0:
            predicted_probas = predicted_probas + abs(min_proba)
        norm_probas = predicted_probas / predicted_probas.sum()
        return norm_probas

    def simulate(self, choice_function=random_choices):
        choosers, alternatives = self.calculate_model_variables()

        choosers, alternatives = self.lcm.apply_predict_filters(
                                         choosers, alternatives)

        choosers = choosers[choosers[self.lcm.choice_column] == -1]
        print("{} agents are making a choice.".format(len(choosers)))

        choices = choice_function(self, choosers, alternatives)
        return choices

    def predict(self, choosers, alternatives, debug=True):
        if len(choosers) == 0:
            return pd.Series()

        if len(alternatives) == 0:
            return pd.Series(index=choosers.index)

        probabilities = self.calculate_probabilities(
                              choosers[[self.lcm.choice_column]],
                              alternatives)

        if debug:
            self.sim_pdf = probabilities

        choices = dcm.unit_choice(
             choosers.index.values,
             probabilities.index.values,
             probabilities.values)
        return choices


def get_model_category_configs():
    """
    Returns dictionary where key is model category name and value is dictionary
    of model category attributes, including individual model config filename(s)
    """
    yaml_configs = orca.get_injectable('yaml_configs')
    model_category_configs = orca.get_injectable('model_structure')['models']

    for model_category, category_attributes in model_category_configs.items():
        category_attributes['config_filenames'] = yaml_configs[model_category]

    return model_category_configs


def create_lcm_from_config(config_filename, model_attributes, summary_alts_xref=None):
    """
    For a given model config filename and dictionary of model category
    attributes, instantiate a SimulationChoiceModel object.
    summary_alts_xref is an optional pd.Series parameter that maps
    alternatives' ids with submarket categories for supply-demand
    price equilibration.
    """
    model_name = config_filename.split('.')[0]
    model = SimulationChoiceModel.from_yaml(
        str_or_buffer=misc.config(config_filename))
    merge_tables = model_attributes['merge_tables'] \
        if 'merge_tables' in model_attributes else None
    agent_units = model_attributes['agent_units'] \
        if 'agent_units' in model_attributes else None
    choice_column = model_attributes['alternatives_id_name'] \
        if model.choice_column is None and 'alternatives_id_name' \
        in model_attributes else None
    equilibration = model_attributes['equilibration'] if 'equilibration' in model_attributes else False
    model.set_simulation_params(model_name,
                                model_attributes['supply_variable'],
                                model_attributes['vacant_variable'],
                                model_attributes['agents_name'],
                                model_attributes['alternatives_name'],
                                choice_column=choice_column,
                                merge_tables=merge_tables,
                                agent_units=agent_units,
                                summary_alts_xref=summary_alts_xref,
                                equilibration=equilibration)
    return model


def create_rm_from_config(config_filename, model_attributes):
    """
    For a given model config filename and dictionary of model category
    attributes, instantiate a SimulationChoiceModel object.
    """
    model_name = config_filename.split('.')[0]
    model = RegressionModel.from_yaml(
        str_or_buffer=misc.config(config_filename))
    dep_var = [dvar for dvar in model_attributes['dep_var'] if
               dvar in config_filename][0]
    model.name = model_name
    model.dep_var = dep_var
    model.observations_name = model_attributes['observations_name']
    return model
