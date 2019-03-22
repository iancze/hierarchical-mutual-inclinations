import numpy as np
from astropy.table import Table
from astropy.io import ascii

deg = np.pi/180.0

# table of stars with inclinations and uncertainties
# name, #i_disk, i_disk_err, i_star, i_star_err (deg)
rows = [("V4046 Sgr", 33.5, 1.4, 33.42, 0.58),
("AK Sco", 109.4, 0.5, 108.76, 2.4),
("DQ Tau", 160.0, 3.0, 158.24, 2.77),
("UZ Tau E", 56.15, 1.5, 56.3, 6.1)]

sample = Table(rows=rows, names=["name", "i_disk", "i_disk_err", "i_star", "i_star_err"])

# calculate these errors in cos(i)
cos_i_disk = np.cos(sample["i_disk"] * deg)
cos_i_disk_err = np.sin(sample["i_disk"] * deg) * sample["i_disk_err"] * deg

cos_i_star = np.cos(sample["i_star"] * deg)
cos_i_star_err = np.sin(sample["i_star"] * deg) * sample["i_star_err"] * deg

cos_sample = Table([sample["name"], cos_i_disk, cos_i_disk_err, cos_i_star, cos_i_star_err], names=["name", "cos_i_disk", "cos_i_disk_err", "cos_i_star", "cos_i_star_err"])

print(cos_sample)

ascii.write(cos_sample, "data/real_sample.ecsv", format="ecsv", overwrite=True)
