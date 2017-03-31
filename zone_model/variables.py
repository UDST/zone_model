import numpy as np
import pandas as pd

import orca
from urbansim.utils import misc
from variable_generators import generators

from zone_model import datasources
geography_base_id = orca.get_injectable('geography_id')

#####################
# ZONE VARIABLES 1
#####################


@orca.column('zones', 'all_zones', cache=True)
def all_zones(zones):
    return pd.Series(np.ones(len(zones)).astype('int32'), index=zones.index)


@orca.column('zones', 'residential_units', cache=False)
def residential_units(zones, residential_units):
    du_by_zone = (residential_units[geography_base_id]
                  .groupby(residential_units[geography_base_id])
                  .size())
    return pd.Series(index=zones.index, data=du_by_zone).fillna(0)


@orca.column('zones', 'households', cache=False)
def zones_households(zones, households):
    hh_by_zone = (households[geography_base_id]
                  .groupby(households[geography_base_id])
                  .size())
    return pd.Series(index=zones.index, data=hh_by_zone).fillna(0)


@orca.column('zones', 'jobs', cache=False)
def zones_jobs(zones, jobs):
    jobs_by_zone = (jobs[geography_base_id]
                    .groupby(jobs[geography_base_id])
                    .size())
    return pd.Series(index=zones.index, data=jobs_by_zone).fillna(0)


@orca.column('zones', 'ln_residential_units', cache=True,
             cache_scope='iteration')
def ln_residential(zones):
    return zones.residential_units.apply(np.log1p)


@orca.column('zones', 'z_id', cache=True)
def z_id(zones):
    return zones.index


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
    return (1 * (households.age_of_head <= 35)
            + 2 * (households.age_of_head > 35)
            * (households.age_of_head <= 60)
            + 3 * (households.age_of_head > 60))


@orca.column('households', 'hhsize3plus', cache=True)
def hhsize3plus(households):
    return (households.persons > 2).astype('int32')


@orca.column('households', 'hhsize2', cache=True)
def hhsize2(households):
    return (households.persons == 2).astype('int32')


@orca.column('households', 'young', cache=True)
def young(households):
    return (households.age_cat == 1).astype('int32')


@orca.column('households', 'middle_age', cache=True)
def middle_age(households):
    return (households.age_cat == 2).astype('int32')


@orca.column('households', 'old', cache=True)
def old(households):
    return (households.age_cat == 3).astype('int32')


@orca.column('households', 'with_child', cache=True)
def with_child(households):
    return (households.children > 0).astype('int32')


@orca.column('households', 'x', cache=True, cache_scope='iteration')
def x(households, zones):
    return misc.reindex(zones.x, households[geography_base_id]).fillna(0)


@orca.column('households', 'y', cache=True, cache_scope='iteration')
def y(households, zones):
    return misc.reindex(zones.y, households[geography_base_id]).fillna(0)


#####################
# JOB VARIABLES
#####################

@orca.column('jobs', 'x', cache=True, cache_scope='iteration')
def x(jobs, zones):
    return misc.reindex(
        zones.x, jobs[geography_base_id]).fillna(0)


@orca.column('jobs', 'y', cache=True, cache_scope='iteration')
def y(jobs, zones):
    return misc.reindex(zones.y, jobs[geography_base_id]).fillna(0)


@orca.column('jobs', 'all_jobs', cache=True)
def all_jobs(jobs):
    return pd.Series(
        np.ones(len(jobs.sector_id)).astype('int32'), index=jobs.index)


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
def job_spaces(zones):
    # Zoning placeholder:  job capacity
    return np.round(zones.total_number_of_jobs * 1.2)


# The ELCM capacity variable
@orca.column('zones', 'vacant_job_spaces', cache=False)
def vacant_job_spaces(zones, jobs):
    return (zones.job_spaces
            .sub(jobs[geography_base_id].value_counts(), fill_value=0))


@orca.column('zones', 'residential_units', cache=True)
def residential_units(zones, residential_units):
    size = residential_units.zone_id.value_counts()
    return pd.Series(data=size, index=zones.index).fillna(0)


# The HLCM capacity variable
@orca.column('zones', 'vacant_residential_units', cache=False)
def vacant_residential_units(zones, households):
    return zones.residential_units.sub(
        households[geography_base_id].value_counts(), fill_value=0)


# The RDPLCM capacity variable
@orca.column('zones', 'ru_spaces', cache=True)
def ru_spaces(zones):
    return np.round(zones.residential_units * 1.2)


@orca.column('zones', 'vacant_ru_spaces', cache=False)
def vacant_ru_spaces(zones, residential_units):
    return (zones.ru_spaces
            .sub(residential_units[geography_base_id].value_counts(),
                 fill_value=0))


@orca.column('zones', 'non_residential_units', cache=True)
def non_residential_units(zones, non_residential_units):
    size = non_residential_units.zone_id.value_counts()
    return pd.Series(data=size, index=zones.index).fillna(0)


# The NRDPLCM capacity variable
@orca.column('zones', 'nru_spaces', cache=True)
def nru_spaces(zones):
    return np.round(zones.non_residential_units * 1.2)


@orca.column('zones', 'vacant_nru_spaces', cache=False)
def vacant_nru_spaces(zones, non_residential_units):
    return (zones.nru_spaces
            .sub(non_residential_units[geography_base_id].value_counts(),
                 fill_value=0))


#######################
#     RESIDENTIAL UNITS
#######################

@orca.column('residential_units', 'x', cache=True, cache_scope='iteration')
def x(residential_units, zones):
    return misc.reindex(
        zones.x, residential_units[geography_base_id]).fillna(0)


@orca.column('residential_units', 'y', cache=True, cache_scope='iteration')
def y(residential_units, zones):
    return misc.reindex(
        zones.y, residential_units[geography_base_id]).fillna(0)


@orca.column('residential_units', 'all_resunits', cache=True)
def all_resunits(residential_units):
    return pd.Series(
        np.ones(len(residential_units.year_built)).astype('int32'),
        index=residential_units.index)

#######################
#     NON-RESIDENTIAL UNITS
#######################


@orca.column('non_residential_units', 'x', cache=True, cache_scope='iteration')
def x(non_residential_units, zones):
    return misc.reindex(
        zones.x, non_residential_units[geography_base_id]).fillna(0)


@orca.column('non_residential_units', 'y', cache=True, cache_scope='iteration')
def y(non_residential_units, zones):
    return misc.reindex(
        zones.y, non_residential_units[geography_base_id]).fillna(0)


@orca.column('non_residential_units', 'all_nonresunits', cache=True)
def all_nonresunits(non_residential_units):
    return pd.Series(np.ones(len(non_residential_units)).astype('int32'),
                     index=non_residential_units.index)
