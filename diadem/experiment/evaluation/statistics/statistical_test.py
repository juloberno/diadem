# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import numpy as np
import pandas as pd
import itertools
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel, ks_2samp

# from: https://stats.stackexchange.com/questions/113602/test-if-two-binomial-distributions-are-statistically-different-from-each-other
# test at confidence level 0.95, todo: add support for other conf. levels
def continuous_paired_ks_2samp(data_bin_1, data_bin_2):
    alpha = 0.05 / 4
    result = ks_2samp(data_bin_1, data_bin_2)
    if result[1] <= alpha:
        return True, result.pvalue
    else:
        return False, result.pvalue

    scipy.stats.ks_2samp(dataset1, dataset2)

def continuous_paired_ttest(data_bin_1, data_bin_2):
    alpha = 0.05/4
    result = ttest_rel(data_bin_1, data_bin_2)
    if result[1] <= alpha:
        return True, result.pvalue
    else:
        return False, result.pvalue


def binomial_paired_mcnemartest(data_bin_1, data_bin_2):
    alpha = 0.05/4
    # from https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

    # build up contigency table assuming data is ordered by the test_idx
    success_1_success_2 = np.count_nonzero(np.logical_and(data_bin_1, data_bin_2))
    failed_1_failed_2 = np.count_nonzero(np.logical_and(np.logical_not(data_bin_1), np.logical_not(data_bin_2)))
    success_1_failed_2 = np.count_nonzero(np.logical_and(data_bin_1, np.logical_not(data_bin_2)))
    failed_1_success_2 = np.count_nonzero(np.logical_and(np.logical_not(data_bin_1), data_bin_2))

    contingency_table = [[ success_1_success_2, success_1_failed_2  ],
                         [ failed_1_success_2 , failed_1_failed_2 ]]

    # otherwise warning in mcnemar function and unmeaningful case
    if (success_1_failed_2+ failed_1_success_2) == 0:
        return False, 1

    # calculate mcnemar test
    result = mcnemar(contingency_table, exact=False)
    if result.pvalue <= alpha:
        return True, result.pvalue
    else:
        return False, result.pvalue

def binomial_paired_group_cochranqtest(**kwargs):
    data = kwargs["data_numpy"]
    # procedure from: https://stats.stackexchange.com/questions/108047/cochrans-q-mcnemar-tests-together
    alpha = 0.05

    result = cochrans_q(data)
    if result.pvalue <= alpha:
        return True, result.pvalue
    else:
        return False, result.pvalue

def continuous_paired_group_repeated_measures_anova(**kwargs):
    data_frame = kwargs["data_frame"]
    dependable_variable = kwargs["dependable_variable"]
    conditions = kwargs["conditions"]

    # make one condition out of multiple, otherwise not supported by AnovaRM
    sLength = len(data_frame[dependable_variable])
    data_frame.loc[:, 'condition'] = pd.Series(np.empty(sLength), index=data_frame.index)
    if isinstance(conditions,list) and len(conditions) > 1:
        for name,group in data_frame.groupby(conditions):
            data_frame.loc[data_frame.groupby(conditions).get_group(name).index,"condition"] = "_".join(name)


    data_frame.drop(columns=conditions)
    # todo: list in conditions not supported map to signle condition required, reduce subject size other wise
    #aovrm = AnovaRM(data_frame, depvar=dependable_variable, subject='test_index', within=conditions)
    aovrm = AnovaRM(data_frame[data_frame["test_index"] < 1000], dependable_variable, 'test_index',
                    within=["condition"], aggregate_func=np.mean)
    res = aovrm.fit()

    print(res)
    # todo: how to read pvalue res.summary()...
    return True, 100


def binomial_unpaired_ztest(data_bin_1, data_bin_2):
    alpha = 0.05 # todo: include
    mean_est_1 = np.mean(data_bin_1)
    mean_est_2 = np.mean(data_bin_2)

    n_data_1 = len(data_bin_1)
    n_data_2 = len(data_bin_1)

    gemetric_mean_1_2 = (n_data_1*mean_est_1 + n_data_2*mean_est_2)/(n_data_1+n_data_2)
    test_statistic_z = ( mean_est_1 - mean_est_2) /  np.sqrt( (gemetric_mean_1_2 * (1 - gemetric_mean_1_2)) * ( 1/n_data_1 + 1/n_data_2 ))

    if test_statistic_z > 1.96 or test_statistic_z < -1.96:
        means_significantly_different = True
    else:
        means_significantly_different = False

    return means_significantly_different, test_statistic_z


def paired_significance_test_pandas(experiment_data, separation_level,comparison_levels, comparison_columns):
    # stat test for averaged scenarios
    test_stat_groups = [separation_level]
    test_stat_groups.extend(comparison_levels)

    temp_columns = []
    temp_columns.append(separation_level)
    temp_columns.extend(comparison_levels)
    temp_columns.extend(comparison_columns)
    stat_test_data = pd.DataFrame(columns=temp_columns)
    for sep_name, sep_group in experiment_data.groupby(separation_level):

        # first group test ---------------------------------------------
        comp_groups = sep_group.groupby(comparison_levels)
        group_sizes, counts= np.unique( comp_groups.size().values, return_counts=True)
        if len(group_sizes) > 1:
            raise ValueError("Comparison group in separation group {} has different group sizes {}".format(sep_group, group_sizes))
        group_size = group_sizes[0]
        num_groups =counts[0]

        for comp_column, test_functions in comparison_columns.items():
            groups_test_data = np.empty((group_size, num_groups))  # rows: N samples, columns: k variables
            group_idx = 0
            for comp_name, comp_group in comp_groups:
                groups_test_data[:,group_idx] = comp_group[comp_column].values
                group_idx += 1

            group_test_function = test_functions[1]
            groups_different, p_value = group_test_function(data_numpy=groups_test_data, data_frame=sep_group, dependable_variable=comp_column, conditions=comparison_levels)
            print("In separation group {} at data cplumn {} significance {} with pvalue {}".format(sep_name, comp_column, groups_different, p_value))

        # then pairwise tests -------------------------------------------
        comp_indices = experiment_data.groupby(separation_level).get_group(sep_name).groupby(comparison_levels).groups.keys()
        comp_indices_combinations = list(itertools.combinations(comp_indices, 2))

        for comp_index_pair in comp_indices_combinations:
            append_dict_0 = {separation_level: sep_name}
            append_dict_1 = {separation_level: sep_name}
            for idx, comp_level in enumerate(comparison_levels):
                append_dict_0[comp_level] = comp_index_pair[0][idx]
                append_dict_1[comp_level] = comp_index_pair[1][idx]

            for comp_column, test_functions in comparison_columns.items():
                grouping = []
                grouping.append(separation_level)
                grouping.extend(comparison_levels)
                if isinstance(comp_index_pair[0], tuple):
                    group1 = (sep_name,) + comp_index_pair[0]
                    group2 = (sep_name,) + comp_index_pair[1]
                else:
                    group1 = (sep_name,) + (comp_index_pair[0],)
                    group2 = (sep_name,) + (comp_index_pair[1],)
                data_1 = experiment_data.groupby(grouping).get_group(
                    group1)[comp_column].values
                data_2 = experiment_data.groupby(grouping).get_group(
                    group2)[comp_column].values

                pairwise_test_function = test_functions[0]
                result, z = pairwise_test_function(data_1,
                                                     data_2)

                append_dict_0[comp_column] = result
                append_dict_1[comp_column] = result

            stat_test_data = stat_test_data.append(append_dict_0, ignore_index=True)
            stat_test_data = stat_test_data.append(append_dict_1, ignore_index=True)

    stat_test_result = stat_test_data.groupby(
    test_stat_groups).all()  # <--- evaluates to true if all pairs in a separation group were true
    return stat_test_result