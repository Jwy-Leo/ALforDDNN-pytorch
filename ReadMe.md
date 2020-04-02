We reimplement the work 
Active Learning for Deep Detection Nerual Network, In ICCV19. Pytorch version.

Now stageed we just use it achieve the selection modules, and using the selection images to train our detector module.

Because that this work utilize the segmentation information to mining the important data.

However, the original version detector isn't good enough to compare with another detector work.

The most worthly work is using the spatial mutual information to leverage the uncertainty of the image.

So we did it as baseline to compare with our work in Master Thesis.
