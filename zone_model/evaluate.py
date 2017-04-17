from __future__ import print_function

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import orca

from zone_model import datasources
from zone_model import variables
from zone_model import models


def correlate(observed, predicted):
    return observed.corr(predicted)


xref_url = 'https://storage.googleapis.com/urbansim/zone_model/zone_summary_xref.csv'  # noqa
summary_geog_xref = pd.read_csv(xref_url)
summary_geog_xref = summary_geog_xref.set_index('zone_id').summary_geog_id

location_choice_models = orca.get_injectable('location_choice_models')
for model_name, model in location_choice_models.items():
    print("Evaluating model {}".format(model_name))
    model.summary_alts_xref = summary_geog_xref

    accuracy_score = model.score()
    print("  Accuracy score is {}".format(accuracy_score))

    rmse = model.score(scoring_function=mean_squared_error, aggregate=True)**.5
    print("  RMSE is {}".format(rmse))

    r2 = model.score(scoring_function=r2_score, aggregate=True)
    print("  R2 is {}".format(r2))

    corr = model.score(scoring_function=correlate, aggregate=True)
    print("  Correlation is {}".format(corr))

    relative_probabilities = pd.Series(model.relative_probabilities())
    print("  Variables by probability influence:")
    print(relative_probabilities.sort_values(ascending=False))
    
