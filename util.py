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


def thresholding_algo(y, lag, threshold, influence):
    """Implementation of algorithm from 
    https://stackoverflow.com/a/22640362/6029703
    requires a citation
    """
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))
