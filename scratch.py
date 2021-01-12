import numpy as np
import matplotlib.pyplot as plt
import lumicks.pylake as lk
from copy import deepcopy
from util import load_estimates, extract_estimates, average_around
from find_unfold import forward_test

file = lk.File("Data/adk5_curve1.h5")
list(file.fdcurves)
fd = file.fdcurves["adk5_curve1"]

ddata = fd.d.data
fdata = fd.f.data
fdata = fdata[ddata > 0]
ddata = ddata[ddata > 0]

print(average_around(fdata, 1000))

handles_model = lk.inverted_odijk("dna_handles") + lk.force_offset("dna_handles")
composite_model = lk.odijk("dna_handles") \
    + lk.inverted_marko_siggia_simplified("protein")
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

print(extract_estimates(fit))
print(forward_test(fit, handles_model, np.linspace(0.3,0.35,100),fdata))

plt.figure()
fit[handles_model].plot()
plt.ylabel('Force [pN]')
plt.xlabel('Distance [$\mu$m]')
#plt.show()
plt.close()

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

plt.figure()
fit[handles_model].plot()
fit[composite_model].plot(independent=np.arange(.26, 40, .1))
plt.ylabel('Force [pN]')
plt.xlabel('Distance [$\mu$m]')
plt.savefig("aaa.png")
plt.close()

plt.scatter(ddata[0:1600],fdata[0:1600], c = 'blue',alpha=0.5)
plt.scatter(ddata[2200:2600],fdata[2200:2600], c = 'orange',alpha=0.5)
print(fit)
fit[handles_model].plot()
fit[composite_model].plot()
plt.savefig("fits.png")
plt.close()

print(fit["dna_handles/Lp"].value, "+-", fit["dna_handles/Lp"].stderr)
print(fit["dna_handles/Lc"].value, "+-", fit["dna_handles/Lc"].stderr)
print(fit["dna_handles/St"].value, "+-", fit["dna_handles/St"].stderr)
xd = np.linspace(0.2, 0.4, 100)
xf = np.linspace(1, 30, 100)

sim1 = handles_model(xd, fit)
sim2 = composite_model(xf, fit) # aaaa

plt.scatter(ddata,fdata, c = 'gray')
plt.scatter(ddata[0:1600],fdata[0:1600], c = 'blue',alpha=0.3)
plt.scatter(ddata[2200:2600],fdata[2200:2600], c = 'orange',alpha=0.3)
plt.plot(xd, sim1)
plt.plot(sim2,xf)
plt.savefig("f2.png")
