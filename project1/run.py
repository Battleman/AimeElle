from proj1_helpers import *
import itertools
from implementations import *
from top_features import top_features
import numpy as np

#load the data
(y, tx, ids) = (np.array(x) for x in load_csv_data("data/train.csv"))

#We quickly saw that the data is split into 4 categories, 
# according to the `PRI_jet_num` column. 
# We split according to these categories

CAT_COL = 22
NUM_CATEGORIES = 4
rows_per_cat = np.array([np.where(tx[:,CAT_COL] == c)[0] for c in range(NUM_CATEGORIES)])

# this represents, for each category, which column contain at least one unknown (-999) value
unknown_cols = [set(np.where(tx[np.where(tx[:, CAT_COL] == c)[0], :] == -999)[1]) for c in range(NUM_CATEGORIES)]

# for i, (rows, cols) in enumerate(zip(rows_per_cat, unknown_cols)):
#     percentages = np.asarray([len(np.where(tx[rows, i] == -999)[0]) \
#                              /len(rows) for i in cols]) * 100
#     print("\nCATEGORY {}:".format(i))
#     for col, perc in zip(cols, percentages):
#         print("Col {} has {}% of unknown".format(col, perc))


# Thus, for every category, any column (except 0) that has any unknown value has only unknown values. 
# This can be explained by the fact that some fields might not be relevant to a certain category, and thus filled with "NaN", or -999
# From now on, for all categories, we only keep columns with values different that -999

# remove col 0 from columns with unknown values.
# We'll deal with that later
for cat in unknown_cols:
    cat.remove(0)

#We remove columns PHI, as they are identically distributed and provide no real meaning.
# We also remove row 22, as data is now split according to it.
# As categories 2 and 3 are basically the same, we consider them as identical

phi_cols = [15, 18, 20, 25, 28]
columns_to_remove = [np.unique(np.concatenate((list(unknown), [CAT_COL], phi_cols))) for unknown in unknown_cols]
columns_to_remove = columns_to_remove[:-1]

NUM_CATEGORIES = 3
rows_per_cat[2] = np.unique(np.concatenate((rows_per_cat[2],rows_per_cat[3])))
rows_per_cat = rows_per_cat[:-1]


## Data cleaning
tx_by_cat = [tx[rows] for rows in rows_per_cat]
y_by_cat = [y[rows] for rows in rows_per_cat]

tx_by_cat = [np.delete(tx_by_cat[cat], np.array(columns_to_remove[cat], dtype=int), axis=1) for cat in range(NUM_CATEGORIES)]

# Replacing remaining NaN by the average value of the column for the category
for cat in tx_by_cat:
    first_col = cat[:,0]
    first_col[first_col == -999] = np.mean(first_col[first_col != -999])

top_feat_by_cat, other_feat_by_cat = top_features(tx_by_cat,y_by_cat, 5)

degrees = [7,10,10]
augmented_tx_by_cat = []
print("Computing features augmentation")
for cat, deg, top_feat, other_feat in zip(tx_by_cat, degrees, top_feat_by_cat, other_feat_by_cat):
    augmented_tx_by_cat.append(build_multinomial(cat, deg, top_feat, other_feat))
    
lambdas = [0.001, 0.003, 0.01]
weights_by_cat = []
print("Computing weights")
for lamb, tx in zip(lambdas, augmented_tx_by_cat, y_by_cat):
    weights_by_cat.append(ridge_regression(y_by_cat, tx_by_cat, lamb))

print("Computing predictions")
    