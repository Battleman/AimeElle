import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

BASEDIR = "figures/"


def y_compare(tx, y, w, filename):
    """
        Plots comparison between actual values and predicted values

        Arguments:
            y {np.ndarray} -- 1D array of actual values
            X {np.ndarray} -- 2D array of samples
            w {np.ndarray} -- 1D array of regression coefficients
            filename {str} -- Name of file to save figure
    """
    y_hat = tx.dot(w)
    plt.scatter(y, y_hat, s=4)
    plt.plot(np.arange(max(y)), np.arange(max(y)), color="black")
    plt.xlabel('y')
    plt.ylabel('ŷ')
    plt.text(2, 180, "r²: {}".format(r2_score(y, y_hat)))
    plt.savefig(BASEDIR+filename)
    plt.show()


def snr_plot(tx, y, w, unc, filename):
    """
        Plot and save SNR graph

        Arguments:
            y {np.ndarray} -- 1D array of actual values
            X {np.ndarray} -- 2D array of samples
            w {np.ndarray} -- 1D array of regression coefficients
            unc {np.ndarray} --  1D array of uncertainty of each `y` value
            filename {str} -- Name of file to save figure
    """
    y_hat = tx.dot(w)
    plt.scatter(y/unc, (y-y_hat)/unc)
    plt.xlabel('SNR')
    plt.ylabel('Residuals normalized')
    plt.savefig(BASEDIR+filename)
    plt.show()


def cross_validation_visualization(lambdas, mse_tr, mse_te, filename):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambdas, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambdas, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(BASEDIR+filename)
    plt.show()

def ridge_regression_lambdas_visualization(lambdas, losses, filename):
    plt.semilogx(lambdas, losses, marker=".", color='b', label='train error')
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.title("Ridge regression")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(BASEDIR+filename)
    plt.show()