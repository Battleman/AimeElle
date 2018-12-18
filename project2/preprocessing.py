
# coding: utf-8

import gc

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures


def get_initial_df(basedir):
    """
        Read the necessary csv files and creates corresponding DataFrames

        Arguments:
            basedir {path} -- Path where to find the csv files.\
            Absolute or relative to the current file.

        Returns:
            pd.DataFrame, pd.DataFrame, pd.DataFrame -- 3 DataFrames,\
            respectively of `spectra`, `measures` and `train_test_split`
    """

    filename_measures = '{}/IMPROVE_2015_measures_cs433.csv'.format(basedir)
    filename_spectra = '{}/IMPROVE_2015_raw_spectra_cs433.csv'.format(basedir)
    filename_tts = '{}/IMPROVE_2015_train_test_split_cs433.csv'.format(basedir)
    # filename_sec_deriv = '/IMPROVE_2015_2nd-derivative_spectra_cs433.csv'\
    # .format(basedir)

    df_spectra_raw = pd.read_csv(filename_spectra)
    df_measures_raw = pd.read_csv(filename_measures)
    df_train_test_split_raw = pd.read_csv(filename_tts)
    # df_second_derivative = pd.read_csv(filename_sec_deriv, index_col=0)
    return df_spectra_raw, df_measures_raw, df_train_test_split_raw


# def get_sample_code(df, col):

def preparation(df_spectra_raw, df_measures_raw, meta_cols, unc_col, y_col):
    """
        Prepares a DataFrame by merging spectra and measures and selecting\
        necessary columns

        Any column of `df_measures_raw`  not in `meta_cols` or `y_col` will be\
        dropped. The other will be merged with respect to the index with\
        `df_spectra_raw`.

        Arguments:
            df_spectra_raw {pd.DataFrame} -- DataFrame containing the spectra\
                (~ the X)
            df_measures_raw {pd.DataFrame} -- DataFrame containing the measures\
                and some meta data (~ the y)
            meta_cols {list} -- List containing the columns of df_measures_raw\
                that are meta data
            y_col {str} -- Name of the column that contain the measures

        Returns:
            pd.DataFrame -- A unique DataFrame containing both measures\
                and spectra.
    """

    df_measures = df_measures_raw.set_index('site')
    df_measures = df_measures[meta_cols + [y_col, unc_col]]
    df_measures.index = pd.Index(df_measures.index, name="")

    df_spectra = df_spectra_raw.T
    df_spectra.columns = pd.Float64Index(
        df_spectra.loc['wavenumber', :], name="")
    df_spectra = df_spectra.drop('wavenumber')

    # ## Dataframes merging
    merged = pd.merge(df_spectra, df_measures,
                      left_index=True, right_index=True)
    nan_indices = merged[y_col].index[merged[y_col].apply(np.isnan)]
    merged.drop(nan_indices, inplace=True)
    return merged


def splitting(merged, df_train_test_split_raw, meta_cols, unc_col, y_col):
    """ Test/train separation"""
    train = df_train_test_split_raw[df_train_test_split_raw.usage ==
                                    "calibration"].site
    test = df_train_test_split_raw[df_train_test_split_raw.usage == "test"].site

    merged_train = merged.loc[np.isin(merged.index, train)]
    merged_test = merged.loc[np.isin(merged.index, test)]

    del train
    del test
    gc.collect()

    # ## X,y creation
    tx_train = merged_train.loc[:, [
        x for x in merged_train.columns if x not in [y_col, unc_col] + meta_cols]]
    y_train = merged_train[y_col]
    unc_train = merged_train[unc_col]
    tx_test = merged_test.loc[:, [
        x for x in merged_test.columns if x not in [y_col, unc_col] + meta_cols]]
    y_test = merged_test[y_col]
    unc_test = merged_test[unc_col]

    return tx_train, y_train, unc_train, tx_test, y_test, unc_test


def features_selection(X, y, k=30, method="f_regression"):

    # ## Features selection
    if method.lower() == "f_regression":
        function = f_regression
    else:
        raise NotImplementedError("Please select another method")
    k_best = SelectKBest(score_func=function, k=k)
    k_best.fit(X, np.ravel(y))
    selected_cols = X.columns[k_best.get_support()]
    return selected_cols


def features_expansion(X, bests_degree, best_cols):
    """
        Expands the "best" columns of X by combining them with others with a\
        maximum total degree of `best_degree`

        Example:
        `best_cols = ['a','c','f']`, and `best_degree = 3`. Then it will\
            compute `X['a']^3, (X['a']^2)(X['c']), X['a']X['c']X['f'], ...`\
            and append them to the current X.

        Arguments:
            X {pd.DataFrame} -- The DataFrame containing the columns to expand
            bests_degree {int} -- The max degree to which expand the columns
            best_cols {list or pd.Index} -- The columns of X that must\
                be expanded

        Returns:
            pd.DataFrame -- X expanded with the polynomial features
    """

    poly_features = PolynomialFeatures(degree=bests_degree,
                                       interaction_only=False,
                                       include_bias=False)
    new_features = pd.DataFrame(poly_features.fit_transform(X[best_cols]),
                                index=X.index)
    mask_best_cols = np.isin(X.columns, best_cols)
    # TODO est-ce que les combinaisons incluent aussi x1, x2,... (aka)
    # est-ce qu'il faut les enlever du DF original ?
    return pd.concat(
        [X[X.columns[~mask_best_cols]], new_features],
        axis=1)
