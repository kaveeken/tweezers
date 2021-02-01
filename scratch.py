import numpy as np
import matplotlib.pyplot as plt
import lumicks.pylake as lk
from copy import deepcopy
from util import load_estimates, extract_estimates, average_around, \
    thresholding_algo
from find_events import run_average, find_turnaround, find_unfold
from build_model import build_model
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


#fnames = ["Data/KPTRmD_marker1.h5",
          #"Data/KPTRmD_marker2.h5"]

#curves = [lk.File(fname) for fname in fnames]
curve = lk.File("Data/TrmD_Marker25.h5")
curve.downsampled_force1x.plot()
curve.force1x.plot()
plt.show()
#plt.savefig("force_1hr.png")
plt.close()

for i in range(5):
    lk.File(f"Data/KPTrmD_Curve{i+1}.h5").fdcurves.get(f"{i+1}").plot_scatter()
plt.show()
#print(curve.distance1)
#plt.scatter(curve.force1x.data[0:10], curve.distance1.data[0:10])

#curve.force1x.plot(start=curve600)
#curve.distance1.plot(start=600)
#plt.show()

#plt.scatter(curve.downsampled_force1x.data[4500:5500], curve.distance1.data[4500:5500])
#plt.show()
#plt.plot(curve.downsampled_force1x.data[5200:5300])
#plt.show()
#curve.downsampled_force1x[4500:].plot()
#plt.show()
#
#curve.fdcurves.get()
fnames = ["Data/adk5_curve1.h5",
          "Data/adk5_curve2.h5",
          "Data/adk5_curve3.h5"]

fds = parse_files(fnames)
dist_datas = [fd.d.data[fd.d.data > 0] for fd in fds]
force_datas = [fd.f.data[fd.d.data > 0] for fd in fds]
peak_indices = [get_peak_indices(f) for f in force_datas]
trough_indices = [get_trough_indices_backwards(f) for f in force_datas]

handles_model = build_model("dna_handles", ["inv_odijk", "force_offset"])
# handles_model = lk.inverted_odijk("dna_handles") \
    # + lk.force_offset("dna_handles")
composite_model_as_func_of_force = lk.odijk("dna_handles") \
    + lk.inverted_marko_siggia_simplified("protein")
composite_model = composite_model_as_func_of_force.invert(interpolate=True,
                                                          independent_min=0,
                                                          independent_max=90) \
                  + lk.force_offset("dna_handles")

#adding inverted inverted M-S simple to handles_model does not work
# composite_model = lk.force_offset("dna_handles") + \
#     (build_model("dna_handles", ["odijk"]) +\
#      build_model("protein",
#                 ["inv_marko_siggia_simple"])).invert(interpolate=True,
#                                                     independent_min = 0,
#                                                     independent_max = 90)

# composite_model = (handles_model.invert() +\
#     build_model("protein",
#                 ["inv_marko_siggia_simple"])).invert(interpolate = True,
#                                                      independent_min = 0,
#                                                      independent_max = 90)
# composite_model = handles_model + build_model("protein", ["marko_siggia_force"])

#composite_model = handles_model + build_model("protein", ["inv_free_jointed_chain"])

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
#fit["protein/Lp"].lower_bound = .6
#fit["protein/Lp"].upper_bound = 1.0
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
plt.show()
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

y = force_datas[0]
x = np.arange(len(force_datas[0]))
def find_transitions(y: np.ndarray, noise_estimation_window: tuple = None):
    EPS = 1e-4  # SNR stabilization factor 
    
    # Magic numbers
    SNR_SCALE_FACTOR = 10
    MIN_OUTLIER_FACTOR = 1.5
    MAX_OUTLIER_FACTOR = 4.5
    MIN_PERCENTILE = 10
    
    # Get noise estimation window
    if noise_estimation_window is None:
        end_slice = max(int(len(y)/10), 3)
        s = slice(0, end_slice)
    else:
        s = slice(**noise_estimation_window)
        
    # Calculate outlier threshold
    snr = (y.max() - y.min()) / (y[s].std() + EPS)
    outlier_factor = min(max(snr/SNR_SCALE_FACTOR, MIN_OUTLIER_FACTOR), MAX_OUTLIER_FACTOR)
    
    # Find outliers that deviate below the threshold (since force transitions are always negative in slope)
    dy = np.diff(y)
    low_percentile = np.nanpercentile(dy, MIN_PERCENTILE)
    median_low_diff = np.nanmedian(dy) - low_percentile
    outlier_threshold = low_percentile - outlier_factor * median_low_diff
    
    return np.where(dy < outlier_threshold)[0], outlier_threshold
breaks, threshold = find_transitions(y)
plt.plot(x,y)
for idx in breaks:
    plt.axvline(x=idx,c="tab:orange")
    
plt.show()
