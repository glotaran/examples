import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.interactive(True)
import numpy as np
import os
# Of course we want Glotaran
import glotaran as gta
from glotaran.plotting.glotaran_color_codes import get_glotaran_default_colors_cycler
from cycler import cycler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'data.ascii')

# Read in streakdata.ascii from resources/data sub-folder
dataset = gta.io.TimeExplicitFile(data_path).read("dataset1")
data = dataset.get()
times = dataset.get_time_axis()
times_shifted = list(np.asarray(times) + 83)
wavelengths = dataset.get_spectral_axis()

# # Get data limits
# if reproduce_figures_from_paper:
#     [xmin, xmax] = [-20, 200] #with respect to maximum of IRF (needs function written)
#     [ymin, ymax] = [630,770]
#     linear_range = [-20, 20]
# else:
#     [xmin,xmax] = [min(dataset_te.get_axis("time")), max(dataset_te.get_axis("time"))]
#     [ymin, ymax] = [min(dataset_te.get_axis("spec")),max(dataset_te.get_axis("spec"))]
#     linear_range = [-20, 20]
# print([xmin,xmax,ymin,ymax])
plt.figure(figsize=(12, 8))
plt.subplot(3, 4, 1)
plt.title('Data')
plt.pcolormesh(times, wavelengths, data)

rsvd, svals, lsvd = np.linalg.svd(data)
plt.subplot(3, 4, 2)
plt.title('LSV Data')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())  # unsure why this is not working
for i in range(4):
    plt.plot(times, lsvd[i, :])
# Plot singular values (SV)
plt.subplot(3, 4, 3)
plt.title('SVals Data')
plt.plot(range(max(10, min(len(times), len(wavelengths)))), svals, 'ro')
plt.yscale('log')
# Plot right singular vectors (RSV, wavelengths, first 3)
plt.subplot(3, 4, 4)
plt.title('RSV Data')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())
for i in range(4):
    plt.plot(wavelengths, rsvd[:, i])

plt.draw()
plt.show()
plt.pause(0.5)

model_spec = '''
type: kinetic
parameters: 
  - initial_concentration:
    - [1, {vary: false}]
    - [0, {vary: false}]
    - [0, {vary: false}]
    - [0, {vary: false}]
  - irf:
    - ["center", -83.0]
    - ["width", 1.5]
    - [13200.0, "backsweep", {vary: false}]
  - kinetic:
    - 0.2
    - 0.07
    - 0.02
    - 0.00016
      
compartments: [s1, s2, s3, s4]

megacomplexes:
    - label: mc1
      k_matrices: [km1]
      
k_matrices:
  - label: "km1"
    matrix: {
      '("s2","s1")': kinetic.1,
      '("s3","s2")': kinetic.2,
      '("s4","s3")': kinetic.3,
      '("s4","s4")': kinetic.4
    }
    
irf:
  - label: irf
    type: gaussian
    center: irf.center
    width: irf.width
    backsweep: True
    backsweep_period: irf.backsweep

initial_concentration: #equal to the total number of compartments
  - label: inputD1
    parameter: [initial_concentration.1, initial_concentration.2, initial_concentration.3, initial_concentration.4] 
    
datasets:
  - label: dataset1
    type: spectral
    initial_concentration: inputD1
    megacomplexes: [mc1]
    path: ''
    irf: irf
'''

model = gta.parse(model_spec)
model.set_dataset("dataset1", dataset)
print(str(model))
result = model.fit()

result.best_fit_parameter.pretty_print()
residual = result.final_residual()

plt.subplot(3, 4, 9)
levels = np.linspace(0, max(dataset.get().flatten()), 10)
cnt = plt.contourf(times, wavelengths, residual, levels=levels, cmap="Greys")
# This is the fix for the white lines between contour levels
for c in cnt.collections:
    c.set_edgecolor("face")
plt.title('Residuals')
plt.show(block=False)

residual_svd = result.final_residual_svd()
# Plot left singular vectors (LSV, times, first 3)
plt.subplot(3, 4, 10)
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())
plt.title('LSV Residuals')
for i in range(3):
    plt.plot(times, residual_svd[2][i, :])
# Plot singular values (SV)
plt.subplot(3, 4, 11)
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())
plt.title('SVals Residuals')
plt.plot(range(min(len(times), len(wavelengths))), residual_svd[1], 'ro')
plt.yscale('log')
# Plot right singular vectors (RSV, wavelengths, first 3)
plt.subplot(3, 4, 12)
plt.title('RSV Residuals')
for i in range(3):
    plt.plot(wavelengths, residual_svd[0][:, i])

spectra = result.e_matrix('dataset1')
spectra = np.asanyarray(spectra)  # TODO: this workaround should not be necessary
plt.subplot(3, 4, 7)
plt.title('EAS')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())
for i in range(spectra.shape[1]):
    plt.plot(wavelengths, spectra[:, i])
plt.subplot(3, 4, 8)
plt.title('norm EAS')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())
for i in range(spectra.shape[1]):
    scale = max(max(spectra[:, i]), abs(min(spectra[:, i])))
    plt.plot(wavelengths, spectra[:, i] / scale)

concentrations = result.c_matrix('dataset1')
plt.subplot(3, 4, 5)
plt.title('Concentrations')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())
plt.plot(times, concentrations[0])

plt.tight_layout()
plt.show(block=True)
