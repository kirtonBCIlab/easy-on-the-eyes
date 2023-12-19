"""
    Classification
    --------------
        Set of classifiers to use with pre-processed data
"""

# Import libraries
import scipy
import sklearn
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def ssvep_cca():
    """ Placeholder for CCA classifier to be used with SSVEP data. """

def ssvep_riemann(epochs:np.ndarray, labels:np.ndarray):
    """ 
        SSVEP Riemmanian geometry classifier


        found here: https://moabb.neurotechx.com/docs/auto_examples/plot_cross_subject_ssvep.html#sphx-glr-auto-examples-plot-cross-subject-ssvep-py
    """

    # Create pipeline
    pipelines_fb = {}
    pipelines_fb["RG+LogReg"] = make_pipeline(
        Covariances(estimator="lwf"),
        TangentSpace(),
        LogisticRegression(solver="lbfgs", multi_class="auto"),
        )
    
    