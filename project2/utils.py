#!/usr/bin/python3
"""This module provides multiple helpers functions"""
import numpy as np
import csv


def batch_iter(tx, y, batch_size, num_batches=1, shuffle=True):
    """
        Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values\
            'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` \
            matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original\
            data messing with the randomness of the minibatches.
        Example of use :
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
            <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y.iloc[shuffle_indices]
        shuffled_tx = tx.iloc[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_tx.iloc[start_index:end_index].values, shuffled_y.iloc[start_index:end_index].values


def save_results(filename, method, w, loss):
    """
        Append results in a csv file

        Arguments:
            filename {str} -- name of the file where to save
            method {str} -- Name of the method for this result
            w {np.ndarray} -- weights
            loss {float} -- Loss of the model
    """

    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow([method, w, loss])
