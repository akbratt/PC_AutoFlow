# PC\_AutoFlow
This is a tool that automatically segments the aortic valve on phase contrast
MRI. It is experimental and not intended for clinical use.


## Dependencies

pytorch
nibabel
pynrrd


## Usage

python train.py --datasource=\*your\_data\_directory\*


## Notes

The model accepts input image volumes and segmentations in nifty and nrrd
formats, respectively. The data directory must contain folders named 'train',
'test', and 'fakes', the latter being the output folder for automated
segmentations. Naming conventions for input volume files are fairly strict, and
must be of the form 'volume-\*i\*Mag.nii', where \*i\* is the index number of
the volume. Similarly, segmentation files must be of the form
'segmentation-\*i\*AV.seg.nrrd'.

