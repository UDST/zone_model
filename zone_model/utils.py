from __future__ import print_function
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


import orca
from urbansim.utils import misc
from urbansim.models import GrowthRateTransition
from urbansim.models import MNLDiscreteChoiceModel


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


def unit_choices(model, choosers, alternatives):
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
    Returns
    -------
    choices : pandas.Series
        Mapping of chooser ID to alternative ID.
    """
    supply_variable, vacant_variable = (model.supply_variable,
                                        model.vacant_variable)

    available_units = alternatives[supply_variable]
    vacant_units = alternatives[vacant_variable]
    # must have positive index
    vacant_units = vacant_units[vacant_units.index.values >= 0]

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

    if len(choosers) > vacant_units.sum():
        print("WARNING: Not enough locations for movers",
              "reducing locations to size of movers for performance gain")
        choosers = choosers.head(vacant_units.sum())

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
            config = yaml.load(f)
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


def register_choice_model_step(model_name, agents_name, choice_function):

    @orca.step(model_name)
    def choice_model_simulate(location_choice_models):
        model = location_choice_models[model_name]

        choices = model.simulate(choice_function=choice_function)

        print('There are {} unplaced agents.'
              .format(choices.isnull().sum()))

        orca.get_table(agents_name).update_col_from_series(
            model.choice_column, choices, cast=True)

    return choice_model_simulate


class SimulationChoiceModel(MNLDiscreteChoiceModel):
    """
    A discrete choice model with parameters needed for simulation.
    Initialize with MNLDiscreteChoiceModel's init parameters or with from_yaml,
    then add simulation parameters with set_simulation_params().

    """
    def set_simulation_params(self, name, supply_variable, vacant_variable,
                              choosers, alternatives, summary_alts_xref=None):
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
        choosers, alternatives = self.calculate_model_variables()

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

    def calculate_probabilities(self, choosers, alternatives):
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
        choosers = orca.get_table(self.choosers).to_frame(columns_used)

        supply_column_names = [col for col in
                               [self.supply_variable, self.vacant_variable]
                               if col is not None]
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
            choosers = choosers.query(self.choosers_predict_filters)

        observed_choices = choosers[self.choice_column]
        predicted_choices = choice_function(self, choosers, alternatives)

        if self.summary_alts_xref is not None:
            observed_choices = observed_choices.map(self.summary_alts_xref)
            predicted_choices = predicted_choices.map(self.summary_alts_xref)

        if aggregate:
            observed_choices = observed_choices.value_counts()
            predicted_choices = predicted_choices.value_counts()

        return scoring_function(observed_choices, predicted_choices)


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


def create_lcm_from_config(config_filename, model_attributes):
    """
    For a given model config filename and dictionary of model category
    attributes, instantiate a SimulationChoiceModel object.
    """
    model_name = config_filename.split('.')[0]
    model = SimulationChoiceModel.from_yaml(
        str_or_buffer=misc.config(config_filename))
    model.set_simulation_params(model_name,
                                model_attributes['supply_variable'],
                                model_attributes['vacant_variable'],
                                model_attributes['agents_name'],
                                model_attributes['alternatives_name'])
    return model
