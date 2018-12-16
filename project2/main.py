"""Main module. This module is ought to be launched to present\
    the various results.
"""

from preprocessing import features_expansion, features_selection, \
    get_initial_df, preparation, splitting
from models import baseline


def main():
    """Main function"""
    print("Reading DataFrames")
    df_spectra, df_measures, df_tts = get_initial_df('data')
    meta_cols = ['SiteCode', 'Date', 'flag',
                 'Latitude', 'Longitude', 'DUSTf:Unc']
    y_col = 'DUSTf:Value'

    print("Merging the data")
    merged = preparation(df_spectra,
                         df_measures,
                         meta_cols,
                         y_col)
    print("Splitting train/test")
    tx_train, y_train, tx_test, y_test = splitting(merged,
                                                   df_tts,
                                                   meta_cols,
                                                   y_col)
    # First compute the baseline
    print("Computing a baseline computation")
    baseline(tx_train, y_train, tx_test, y_test, "results.csv")

    # best_features = features_selection(tx_train, y_train, 30)
    # tx_train = features_expansion(tx_train, 4, best_features)


if __name__ == "__main__":
    main()
