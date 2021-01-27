import numpy as np
import matplotlib.pyplot as plt
import lumicks.pylake as lk
from copy import deepcopy
from util import load_estimates, extract_estimates, average_around, \
    thresholding_algo
from find_events import run_average, find_turnaround, find_unfold
from sklearn.cluster import KMeans
from scipy.stats import gmean


def parse_files(fnames):
    files = [lk.File(fname) for fname in fnames]
    fds = []
    for i in range(len(fnames)):
        fds.append(files[i].fdcurves[list(files[i].fdcurves)[-1]])
    return fds


def get_peak_indices(f):
    stds = []
    for i in range(25, len(f) - 25):
        std = average_around(f, i, half_n=25)["std"]
        stds.append(std)
    peaksign = thresholding_algo(stds, 500, 4., 0)
    return np.arange(25, len(stds) + 25)[peaksign["signals"] >= 1]


def get_trough_indices_backwards(f):
    stds = []
    for i in range(25, len(f) - 25):
        std = average_around(f, i, half_n=25)["std"]
        stds.insert(0, std)
    peaksign = thresholding_algo(stds, 500, 4., 0)
    backwards_indices = np.arange(25, len(stds) + 25)[peaksign["signals"]
                                                      <= -1]
    return sorted([len(f) - index for index in backwards_indices])


fnames = ["Data/adk5_curve1.h5",
          "Data/adk5_curve2.h5",
          "Data/adk5_curve3.h5"]

fds = parse_files(fnames)
dist_datas = [fd.d.data[fd.d.data > 0] for fd in fds]
force_datas = [fd.f.data[fd.d.data > 0] for fd in fds]
peak_indices = [get_peak_indices(f) for f in force_datas]
trough_indices = [get_trough_indices_backwards(f) for f in force_datas]

handles_model = lk.inverted_odijk("dna_handles") \
    + lk.force_offset("dna_handles")
composite_model_as_func_of_force = lk.odijk("dna_handles") \
    + lk.inverted_marko_siggia_simplified("protein")
composite_model = composite_model_as_func_of_force.invert(interpolate=True,
                                                          independent_min=0,
                                                          independent_max=90) \
                  + lk.force_offset("dna_handles")

fit = lk.FdFit(handles_model, composite_model)
for i in range(len(force_datas)):
    fit[handles_model].add_data(f"closed_{i + 1}",
                                force_datas[i][0:peak_indices[i][0] - 200],
                                dist_datas[i][0:peak_indices[i][0] - 200])

fit["dna_handles/Lc"].value = .35
fit["dna_handles/Lp"].value = 15
fit["dna_handles/St"].value = 300
fit["dna_handles/St"].lower_bound = 250
fit["dna_handles/f_offset"].upper_bound = 6
fit["dna_handles/f_offset"].lower_bound = -6
fit.fit()
print(fit)
# fit[handles_model].plot()
# plt.show()
for i in range(len(force_datas)):
    fit[composite_model].add_data(f"open_{i + 1}",
                                  force_datas[i][trough_indices[i][-1] + 100:
                                                 trough_indices[i][-1] + 600],
                                  dist_datas[i][trough_indices[i][-1] + 100:
                                                trough_indices[i][-1] + 600])

fit["protein/Lp"].value = .7
fit["protein/Lp"].lower_bound = .6
fit["protein/Lp"].upper_bound = 1.0
fit["protein/Lp"].fixed = False
fit["protein/Lc"].value = .01

fit["dna_handles/St"].fixed = True
fit["dna_handles/Lp"].fixed = True
fit["dna_handles/Lc"].fixed = True
fit.fit()

print(fit)
plt.figure()
fit[handles_model].plot()
fit[composite_model].plot(independent=np.arange(.26, .4, .001))
plt.ylabel('Force [pN]')
plt.xlabel('Distance [$\mu$m]')
plt.savefig("fits.png")
# plt.show()
plt.close()

# print(fit.log_likelihood())

stds = [average_around(force_datas[0], i, half_n=25)["std"]
        for i in range(25, len(force_datas[0]) - 25)]
peaks = thresholding_algo(stds, 500, 4., 0)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.plot(force_datas[0][25:-25], label="force")
fig.legend()
plt.savefig("force.png")
ax2.plot(stds, c="tab:orange", label="stdev(force) over period 50")
fig.legend()
plt.savefig("force+sd.png")
ax3.plot(peaks["signals"], c="tab:red", label="peak signal")
fig.legend()
plt.savefig("force+sd+signal.png")
# plt.show()
plt.close()
