import numpy as np
import lumicks.pylake as lk
from util import load_estimates, extract_estimates, average_around

def run_average(data, N):
    cumsum = np.cumsum(data)
    prep = np.repeat(data[0], N)
    averages = (cumsum[N:] - cumsum[:-N]) / N
    return np.concatenate([prep, averages])

def find_turnaround(data, N):
    r_avg = run_average(data, N)
    for i in range(N,len(r_avg)):
        if r_avg[i] < r_avg[i - 1]:
            return i

def get_leg_length(f):
    for i in range(len(f)):
        if force >= max(f) / 4:
            return i

def forward_test(fit, model, landing, f):
    sim = model(landing, fit)
    if sim.std() <= fit.sigma[0]: # wrong
        return False
    else:
        return True

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
