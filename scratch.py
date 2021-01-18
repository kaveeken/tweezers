import numpy as np
import matplotlib.pyplot as plt
import lumicks.pylake as lk
from copy import deepcopy
from util import load_estimates, extract_estimates, average_around
from find_events import run_average, find_turnaround, find_unfold

file = lk.File("Data/adk5_curve1.h5")
list(file.fdcurves)
fd = file.fdcurves["adk5_curve1"]

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

model2 = handles_model
fit[handles_model].add_data("closed", fdata[0:1600],
                            ddata[0:1600])
fit["dna_handles/Lc"].value = .35
fit["dna_handles/Lp"].value = 15
fit["dna_handles/St"].value = 300
fit["dna_handles/St"].lower_bound = 250
fit["dna_handles/f_offset"].upper_bound = 6
fit["dna_handles/f_offset"].lower_bound = -6
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


fit[composite_model].add_data("open", fdata[2200:2600],
                            ddata[2200:2600])

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

unf = find_unfold(fdata[0:2000], ddata[0:2000], handles_model, estimates)

print(unf)
plt.scatter(ddata[0:1600],fdata[0:1600])
plt.scatter(ddata[1600:2000],fdata[1600:2000],c="tab:red")
plt.scatter(ddata[2000:2800],fdata[2000:2800],c="tab:orange")
plt.scatter(ddata[2800:3200],fdata[2800:3200],c="tab:green")
plt.scatter(ddata[3200:3500],fdata[3200:3500],c="tab:blue")
plt.axvline(x=ddata[unf])
plt.scatter(ddata[0:2000],handles_model(ddata[0:2000], fit))
plt.savefig("scatter.png")
plt.show()

sim = handles_model(0.2,fit)
print(sim)


