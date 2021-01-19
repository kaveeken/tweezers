import numpy as np
import matplotlib.pyplot as plt
import lumicks.pylake as lk
from copy import deepcopy
from util import load_estimates, extract_estimates, average_around, thresholding_algo
from find_events import run_average, find_turnaround, find_unfold
from sklearn.cluster import KMeans
from scipy.stats import gmean

file = lk.File("Data/adk5_curve2.h5")
list(file.fdcurves)
fd = file.fdcurves["adk5_curve2"]

ddata = fd.d.data
fdata = fd.f.data
fdata = fdata[ddata > 0]
ddata = ddata[ddata > 0]

print(average_around(fdata, 1000))

handles_model = lk.inverted_odijk("dna_handles") + lk.force_offset("dna_handles")
composite_model_as_function_of_force = lk.odijk("dna_handles") \
    + lk.inverted_marko_siggia_simplified("protein")
composite_model = composite_model_as_function_of_force.invert(interpolate=True,
                                                              independent_min=0,
                                                              independent_max=90) \
                                  + lk.force_offset("dna_handles")
fit = lk.FdFit(handles_model, composite_model)
fit_tmp = lk.FdFit(handles_model, composite_model)
fit_tmp[handles_model].add_data("closed", fdata[0:2000],
                            ddata[0:2000])
fit_tmp["dna_handles/Lc"].value = .35
fit_tmp["dna_handles/Lp"].value = 15
fit_tmp["dna_handles/St"].value = 300
fit_tmp["dna_handles/St"].lower_bound = 250
fit_tmp["dna_handles/f_offset"].upper_bound = 6
fit_tmp["dna_handles/f_offset"].lower_bound = -6
estimates = extract_estimates(fit_tmp)

unf = find_unfold(fdata[0:2000], ddata[0:2000], handles_model, estimates)
print(unf)

model2 = handles_model
fit[handles_model].add_data("closed", fdata[0:unf[0]],
                            ddata[0:unf[0]])
load_estimates(fit, estimates)
fit.fit()

estimates = extract_estimates(fit)

print(extract_estimates(fit))
print("#############################################~")
fit2 = lk.FdFit(handles_model, composite_model)
fit2[handles_model].add_data("closed", fdata[0:1600],
                            ddata[0:1600])
print(fit2)
load_estimates(fit2, extract_estimates(fit))
print(extract_estimates(fit2))

plt.figure()
fit[handles_model].plot()
plt.ylabel('Force [pN]')
plt.xlabel('Distance [$\mu$m]')
#plt.show()
plt.close()
print(fit)


unf_size = 800
fit[composite_model].add_data("open", fdata[unf[1]:unf[1] + unf_size],
                              ddata[unf[1]:unf[1] + unf_size])

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
fit[handles_model].plot()
fit[composite_model].plot(independent=np.arange(.26, .4, .001))
plt.savefig("fits.png")
plt.close()

print(fit["dna_handles/Lp"].value, "+-", fit["dna_handles/Lp"].stderr)
print(fit["dna_handles/Lc"].value, "+-", fit["dna_handles/Lc"].stderr)
print(fit["dna_handles/St"].value, "+-", fit["dna_handles/St"].stderr)

plt.plot(ddata)
plt.plot(run_average(ddata,100))
plt.plot(run_average(ddata,200))
plt.axvline(x=find_turnaround(ddata,200))
#plt.show()
plt.close()

stds = []
for i in range(25, len(ddata) - 25):
    std = average_around(fdata, i, half_n = 25)["std"]
    stds.append(std)

plt.scatter(ddata[0:1600],fdata[0:1600])
plt.scatter(ddata[unf[1]:unf[1] + unf_size],fdata[unf[1]:unf[1] + unf_size],
            c="tab:orange")
plt.scatter(ddata[unf[1] + unf_size:],fdata[unf[1] + unf_size:],
            c="tab:green")
#plt.scatter(ddata[3200:3500],fdata[3200:3500],c="tab:blue")
plt.scatter(ddata[unf[0]:unf[1]],fdata[unf[0]:unf[1]],c="tab:red")
plt.axvline(x=ddata[unf[0]])
plt.axvline(x=ddata[unf[1]])
plt.axvline(x=ddata[unf[0] + np.argmax(stds)], c="red")
plt.plot(ddata,handles_model(ddata, fit), c="black")
plt.plot(ddata,composite_model(ddata, fit), c="black")
plt.scatter(ddata[25:-25][peaked["signals"] >= 1],
            fdata[25:-25][peaked["signals"] >= 1], c="tab:purple")
plt.ylim(bottom=0,top=50)
plt.xlim(left=0.26,right=0.42)
plt.savefig("scatter.png")
#plt.show()
plt.close()

plt.plot(ddata)
plt.show()

plt.plot(stds)
plt.axhline(y=gmean(stds),c="tab:orange")
plt.axhline(y=np.mean(stds),c="tab:blue")
plt.axhline(y=np.median(stds),c="tab:green")
#plt.axhline(y=gmean(stds) + np.std(stds),c="tab:orange")
plt.axhline(y=np.mean(stds) + np.std(stds),c="tab:blue", alpha=0.5)
plt.axhline(y=np.mean(stds) - np.std(stds),c="tab:blue", alpha=0.5)
plt.show()

peaked = thresholding_algo(stds, 500, 3.6, 0)
plt.plot(peaked["signals"], c="red")
plt.plot(peaked["avgFilter"], c="tab:blue")
#plt.show()
plt.close()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(stds)
ax2.plot(peaked["signals"], c="red")
plt.show()
