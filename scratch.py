import numpy as np
import matplotlib.pyplot as plt
import lumicks.pylake as lk
import h5py

d = lk.File('Data/20210302:190729 Marker 4_TrmD.h5')
plt.plot(d.distance1.data)#, d.downsampled_force1.data[1000:1200], '.')
plt.show()
#print(d.fdcurves)
#stamps = [1606227339070643800, 1606227343599143000]
#fdc = lk.fdcurve.FDCurve(d, stamps[0], stamps[1], 'test')
#print(fdc.d.data)
#d.fdcurves.get('test')
#dd = h5py.File('Data/TrmD_marker24.h5')

d2 = lk.File('Data/adk5_curve1.h5')
