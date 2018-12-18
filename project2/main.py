"""Main module. This module is ought to be launched to present\
    the various results.
"""

import gc

import numpy as np

from models import baseline, gradient_descent
from preprocessing import (features_expansion, features_selection,
                                    get_initial_df, preparation, splitting)
from visualization import snr_plot, y_compare

OUTPUT_FILE = "results.csv"


def main():
    """Main function"""
    print("Reading DataFrames")
    df_spectra, df_measures, df_tts = get_initial_df('data')
    meta_cols = ['SiteCode', 'Date', 'flag',
                 'Latitude', 'Longitude']
    unc_col = 'DUSTf:Unc'
    y_col = 'DUSTf:Value'

    print("Merging the data")
    merged = preparation(df_spectra,
                         df_measures,
                         meta_cols,
                         unc_col,
                         y_col)
    del df_spectra
    del df_measures
    print("Splitting train/test")
    tx_train, y_train, _, tx_test, y_test, unc_test = splitting(merged,
                                                                df_tts,
                                                                meta_cols,
                                                                unc_col,
                                                                y_col)
    del merged
    del df_tts
    gc.collect()
    gc.collect()
    # First compute the baseline
    gd_initial_weights = np.random.rand(tx_train.shape[1])

    #########################
    # BASELINE
    #########################
    # print("Computing a baseline computation")
    # loss, weights = baseline(tx_train, y_train, tx_test, y_test, OUTPUT_FILE)
    # print("\tloss of {}".format(loss))
    # y_compare(tx_test, y_test, weights, "baseline_y-compare_simple.png")
    # snr_plot(tx_test, y_test, weights, unc_test, "baseline_snr_simple.png")

    #########################
    # GRADIENT
    #########################
    print("Computing a gradient descent")
    weights = gradient_descent(tx_train, y_train,
                               gd_initial_weights, 50, 0.005)
    y_compare(tx_test, y_test, weights, "GD_y-compare_simple.png")
    snr_plot(tx_test, y_test, weights, unc_test, "GD_snr_simple.png")

    #########################
    # EXPANSION
    #########################
    print("Now expanding features")
    best_features = features_selection(tx_train, y_train, 30)
    tx_train_expanded = features_expansion(tx_train, 4, best_features)
    tx_test_expanded = features_expansion(tx_test, 4, best_features)

    #########################
    # BASELINE EXPANDED
    #########################
    print("Computing baseline with expanded features")
    loss, weights = baseline(tx_train_expanded, y_train,
                             tx_test_expanded, y_test,
                             OUTPUT_FILE)
    print("\tloss of {}".format(loss))
    y_compare(tx_test_expanded, y_test, weights,
              "baseline_y-compare_expanded.png")
    snr_plot(tx_test_expanded, y_test, weights,
             unc_test, "baseline_snr_expanded.png")

    #######################
    # GRADIENT EXPANDED
    #######################
    print("Computing a gradient descent with expanded features")
    loss, weights = gradient_descent(tx_train_expanded, y_train,
                                     gd_initial_weights, 50, 0.005)
    print("\tloss of {}".format(loss))
    y_compare(tx_test_expanded, y_test, weights, "GD_y-compare_simple.png")
    snr_plot(tx_test_expanded, y_test, weights,
             unc_test, "baseline_snr_expanded.png")


if __name__ == "__main__":
    main()
