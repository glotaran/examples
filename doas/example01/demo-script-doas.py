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

dataset = gta.io.WavelengthExplicitFile("HIBA1172DY_2.ascii").read("dataset1")
data = dataset.get()
time_axis = dataset.get_time_axis()
spectral_axis = dataset.get_spectral_axis()
