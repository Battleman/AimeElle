
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spstats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_regression, SelectKBest
import gc


def main():
    df_spectra_raw, df_measures_raw, df_train_test_split_raw = get_initial_df(
        'data')
    meta_cols = ['SiteCode', 'Date', 'flag',
                 'Latitude', 'Longitude', 'DUSTf:Unc']
    y_col = ['DUSTf:Value']

    merged = preparation(df_spectra_raw,
                         df_measures_raw,
                         df_train_test_split_raw,
                         meta_cols,
                         y_col)
    X, y, X_test, y_test = splitting(merged,
                                     df_train_test_split_raw,
                                     meta_cols,
                                     y_col)
    best_features = features_selection(X, y, 30)
    X = features_expansion(X, 4, best_features)
    print(X)

def get_initial_df(basedir):
    filename_measures = '{}/IMPROVE_2015_measures_cs433.csv'.format(basedir)
    filename_spectra = '{}/IMPROVE_2015_raw_spectra_cs433.csv'.format(basedir)
    filename_tts = '{}/IMPROVE_2015_train_test_split_cs433.csv'.format(basedir)
    # filename_sec_deriv = '/IMPROVE_2015_2nd-derivative_spectra_cs433.csv'.format(basedir)

    df_spectra_raw = pd.read_csv(filename_spectra)
    df_measures_raw = pd.read_csv(filename_measures)
    df_train_test_split_raw = pd.read_csv(filename_tts)
    # df_second_derivative = pd.read_csv(filename_sec_deriv, index_col=0)
    return df_spectra_raw, df_measures_raw, df_train_test_split_raw


def preparation(df_spectra_raw, df_measures_raw, meta_cols, y_col):

    df_measures = df_measures_raw.set_index('site')
    df_measures = df_measures[meta_cols + y_col]
    df_measures.index = pd.Index(df_measures.index, name="")

    df_spectra = df_spectra_raw.T
    df_spectra.columns = pd.Float64Index(
        df_spectra.loc['wavenumber', :], name="")
    df_spectra = df_spectra.drop('wavenumber')

    # ## Dataframes merging
    merged = pd.merge(df_spectra, df_measures,
                      left_index=True, right_index=True)
    nan_indices = merged[y_col[0]].index[merged[y_col[0]].apply(np.isnan)]
    merged.drop(nan_indices, inplace=True)
    return merged


def splitting(merged, df_train_test_split_raw, meta_cols, y_col):
    """ Test/train separation"""
    train = df_train_test_split_raw[df_train_test_split_raw.usage ==
                                    "calibration"].site
    test = df_train_test_split_raw[df_train_test_split_raw.usage == "test"].site
    merged_train = merged.loc[np.isin(merged.index, train)]
    merged_test = merged.loc[np.isin(merged.index, test)]

    del(train)
    del(test)
    gc.collect()

    # ## X,y creation
    X_train = merged_train.loc[:, [
        x for x in merged_train.columns if x not in y_col and x not in meta_cols]]
    y_train = merged_train[y_col]
    X_test = merged_train.loc[:, [
        x for x in merged_test.columns if x not in y_col and x not in meta_cols]]
    y_test = merged_train[y_col]

    return X_train, y_train, X_test, y_test


def features_selection(X, y, k=30):
    # ## Features selection
    test = SelectKBest(score_func=f_regression, k=k)
    test.fit(X, np.ravel(y))
    selected_cols = X.columns[test.get_support()]
    return selected_cols


def features_expansion(X, bests_degree, best_cols):
    pf = PolynomialFeatures(degree=bests_degree,
                            interaction_only=False, include_bias=False)
    new_features = pd.DataFrame(pf.fit_transform(X[best_cols]), index=X.index)
    return pd.concat([X[X.columns[~np.isin(X.columns, best_cols)]], new_features], axis=1)
