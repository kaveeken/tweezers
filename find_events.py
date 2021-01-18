import numpy as np
import lumicks.pylake as lk
from util import load_estimates, extract_estimates, average_around
from matplotlib import pyplot as plt

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
        if f[i] >= max(f) / 4:
            return round(i / 2)


def sort_by_dist(f, d):
    d_sort = sorted(d) # maybe not sort twice
    f_sort = [force for dist, force in sorted(zip(d,f))]
    return f_sort, d_sort


def search_unfold(f,d,model,est,leg_length,leg_last):
    fit = lk.FdFit(model)
    fit[model].add_data("prefold", f[0:leg_last], d[0:leg_last])
    load_estimates(fit,est)
    fit.fit()

    target = leg_last + leg_length
    sim = model(d[0:target], fit)
    plt.scatter(d[0:leg_last], f[0:leg_last])
    plt.scatter(d[leg_last:target], f[leg_last:target], c="tab:orange")
    plt.scatter(d[0:target], sim, c="tab:green")
    plt.show()
    force_mean = average_around(f, target)["mean"]
    if force_mean > sim[-1] + 2 * fit.sigma[0]:
        # this is a strange case and probably unlikely
        # - this actually happens a lot b/c test being too stringent
        print("F much larger than simulated")
        print(target)
        return False
        #return 1
    elif force_mean < sim[-1] - 2 * fit.sigma[0]:
        first = target
        for i in reversed(range(leg_last, target)):
            if average_around(f, i)["mean"] < sim[i] -2 * fit.sigma[0]:
                first = i
        return round((first + target) / 2)
    else:
        return False
    

def find_unfold(f, d, model, est):
    leg_length = get_leg_length(f)
    leg_last = leg_length
    found = False
    while not found:
        found = search_unfold(f,d,model,est,leg_length,leg_last)
        leg_last += leg_length
        if leg_last > len(f):
            found = -2

    if found < 0: # this needs to be proper error handling
        print("something went wrong:")
        if found == -1:
            print("something strange")
        elif found == -2:
            print("no unfolding event found")

    return found
