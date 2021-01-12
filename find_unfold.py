import numpy as np
import matplotlib.pyplot as plt
import lumicks.pylake as lk
from copy import deepcopy

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

def average_around(index, data, half_n = 10):
    # this needs some sanity checks around the indexing
    subset = data[index - half_n : index + half_n]
    return {"mean" : np.mean(subset), "std" : np.std(subset)}

def get_leg_length(f):
    for i in range(len(f)):
        if force >= max(f) / 4:
            return i

def forward_test(fit, model, landing, f):
    sim = model(landing, fit)
    if sim.std() <= fit.sigma:
        return False

def find_unfold(f, d, model, est):
    #fit = lk.FdFit(model)
    leg_length = get_leg_length(f)
    leg_last = leg_length
    searching = True
    while searching:
        fit = lk.FdFit(model)
        fit[model].add_data("prefold" , f[0:leg_last], d[0:leg_last])
        load_estimates(fit, est)
        fit.fit()
        leg_last += leg_length
        landing = np.linspace(average_around(leg_last - 100, d)["mean"],
                              average_around(leg_last, d)["mean"], 100)

file = lk.File("Data/adk5_curve1.h5")
list(file.fdcurves)
fd = file.fdcurves["adk5_curve1"]
