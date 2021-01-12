import numpy as np
import matplotlib.pyplot as plt
import lumicks.pylake as lk

def load_estimates(fit, est):
    for key in est["vals"]:
        fit[key].value = est["vals"][key]
    for key in est["bounds_up"]:
        fit[key].upper_bound = est["bounds_up"][key]
    for key in est["bounds_down"]:
        fit[key].lower_bound = est["bounds_down"][key]
    for key in est["fixed"]:
        fit[key].fixed = est["fixed"][key]

def extract_estimates(fit):
    # this dict should probably be structured the same as fit.params
    est = {"vals" : {}, "bounds_up" : {}, "bounds_down" : {}, "fixed" : {}}
    for key in fit.params.keys:
        est["vals"][key] = fit.params[key].value
        est["bounds_up"][key] = fit.params[key].upper_bound
        est["bounds_down"][key] = fit.params[key].lower_bound
        est["fixed"][key] = fit.params[key].fixed
    return est

# def something to write estimates to file

def average_around(data, index, half_n = 10):
    # this needs some sanity checks around the indexing
    subset = data[index - half_n : index + half_n]
    return {"mean" : np.mean(subset), "std" : np.std(subset)}
