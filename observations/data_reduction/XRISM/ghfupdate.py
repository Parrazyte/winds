from astropy.io import fits
from astropy.table import Table, vstack


'''
Mail from Misaki
When observing a bright source like this, the exposure time for the Fe-55 filter should be increased when acquiring the gain, if the source is within the field of view. (I was the one who set the threshold values related to this.) However, since this was a DDT observation, I assume such preparation could not be made in advance.
In the pixel 17 at T=40-45ks, the number of photons from the source was high, and as a result, sufficient Mn Kα events with the Hp grade were not obtained. The rslgain tool is used to calculate the effective temperature for each pixel, and by default, minevent=200 is set, meaning that if fewer than 200 Hp events are detected, Mn Kα line fitting will not be performed. By loosening this constraint, we can recover the missing effective temperatures and save them properly.

Please run the following commands. The required Python script is attached.
------
rslgain infile=xa901002010rsl_p0px5000_cl.evt.gz outfile=Fe55.ghf linetocorrect=MnKa calmethod=Fe55 clobber=yes debug=yes gtifile='xa901002010rsl_p0px5000_cl.evt.gz[2]' spangti=no ckrisetime=yes calcerr=yes writeerrfunc=yes extraspread=40 numevent=1000 minevent=20 maxdshift=10

# you need xa901002010rsl_000_fe55.ghf.gz in your working directory.
python ghfupdate.py  

rslpha2pi infile=xa901002010rsl_p0px1000_cl.evt.gz outfile=xa901002010rsl_p0px1000_cl_update.evt driftfile=fe55_update.ghf clobber=yes
------

After this correction, you can use xa901002010rsl_p0px1000_cl_update.evt instead of xa901002010rsl_p0px1000_cl.evt.gz.

You can explain this in the paper as follows:
------
The energy scale is usually calibrated using the gain history file during the standard pipeline processing. However, in this observation, since the calibration source was irradiated while the target was within the field of view, some pixels did not get a sufficient number of Mn Kalpha photons with the Hp grade. As a result, the default gain history file contained missing values. To address this, we recalculated the gain correction by relaxing the minimum photon count requirement for Mn Kalpha line fitting, and regenerated the cleaned event file using the updated gain information.
------

#TBD: make the two other commands in python
'''

file_ghf_init=input('name of old calibration file (e.g. xa901002010rsl_000_fe55.ghf)')
file_ghf_up=input('name of output of the new run of rslgain (e.g. Fe55.ghf')
file_ghf_out=input('final output gain file name')

# Open source FITS files
with fits.open(file_ghf_init) as a_hdul, fits.open(file_ghf_up) as b_hdul:
    # Convert first extensions to Table objects
    a_table = Table(a_hdul[1].data)
    b_table = Table(b_hdul[1].data)

    # Extract the 122nd row from B (index 121)
    row_to_insert = b_table[121:122]  # slicing keeps it as Table

    # Split A into two parts and insert the row between 103rd and 104th (index 103)
    upper = a_table[:103]
    lower = a_table[103:]
    updated_table = vstack([upper, row_to_insert, lower])

    # Create a new BinTableHDU with the original header
    new_hdu = fits.BinTableHDU(data=updated_table, header=a_hdul[1].header)

    # Combine primary HDU and modified extension
    new_hdul = fits.HDUList([a_hdul[0], new_hdu])

    # Write to new output file
    new_hdul.writeto(file_ghf_out, overwrite=True)
