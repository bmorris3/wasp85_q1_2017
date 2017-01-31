from glob import glob
from astroscrappy import detect_cosmics
from astropy.io import fits
import os
from astropy.utils.console import ProgressBar

# image_paths = sorted(glob('/Users/bmmorris/data/Q1UW01/UT170123/wasp85.????.fits'))[360:]
# outpath = '/Users/bmmorris/data/Q1UW01/UT170123/cleaned/'

image_paths = sorted(glob('/Users/bmmorris/data/Q1UW01/UT170131/wasp85-diffuser.????.fits'))
outpath = '/Users/bmmorris/data/Q1UW01/UT170131/cleaned/'


with ProgressBar(len(image_paths)) as bar:
    for path in image_paths:
        bar.update()
        f = fits.open(path)
        file_name = path.split(os.sep)[-1]
        mask, cleaned_image = detect_cosmics(f[0].data)
        f[0].data = cleaned_image
        fits.writeto(outpath + file_name, cleaned_image, header=f[0].header,
                     clobber=True)
