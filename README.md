# apples2droids

![logo](docs/logo.jpeg "logo")


## About
When downloading Apple photos from ICloud, all Live Photos (images that move for ~3seconds) are downloaded as one static image and one ~3s video. If you have thousands of photos, this can be unbelievably annoying.

If you're migrating away from Apple, you may want to access your Live Photos, but are unable to.

That's why I created **apples2droid**, which comverts Live Photos into one single .jpg, which is compatible with Google's Motion Photo and other Android Motion Photos. AND when you save the new .jgp image elsewhere, it will only be ONE file, not split into two files (static + video) again.

The new .jpg image will remain a Motion Photo, but will only be accessible as a Motion Photo if on an compatible device (eg android device). So saving the image elsewhere won't strip its Motion feature. 

**NOTE**: This code assumes that the Live Photos are split into an image file and a video file (you can manually set which formats you want considered, see **Format Help** section), with identical names - this is standard from ICloud. If they do not have the same name, eg IMG_0001.HEIC and IMG_0001.MOV, then the code will not consider them. 

## Quick Start
You need Python. Make virtual env, then run:

```bash
pip install -r requirements_python.txt
```

**NOTE**: You need **Exiftool** downloaded, see here for all operating systems: [Website](https://exiftool.org/).

### Windows:

Download Exiftools here: [Website](https://exiftool.org/), or download directly: [Windows 64](https://sourceforge.net/projects/exiftool/files/exiftool-13.48_64.zip/download), [Windows 32](https://sourceforge.net/projects/exiftool/files/exiftool-13.48_32.zip/download)


```bash
python apples2droids.py
```


### Linux:

**Debian/Ubuntu**:
```bash
sudo apt install exiftool
```

Exiftool for **Other Linux distros**: [Website](https://exiftool.org/)

```bash
chmod +x apples2droids.py
./apples2droids.py
```

### MacOS:

Download Exiftools here: [Website](https://exiftool.org/)

```bash
chmod +x apples2droids.py
./apples2droids.py
```


## Overview and Usage
The code launches a lightweight front-end `tkinter` GUI to choose the input and output folders, and to execute the code.
**NOTE**: Make sure your output folder is empty, as it will overwrite files of the same.
**NOTE**: The input file is read-only so your images won't be edited or tampered with. BUT PLEASE MAKE SURE YOU ALWAYS HAVE A BACKUP. IF YOUR LAPTOP DIES OR BLUESCREENS AT THE EXACT POINT THE CODE IS READING THE IMAGE, IT COULD CORRUPT IT OR SIMILAR (not that this will likely happen, but always be safe, especially if they're important photos).

The metadata of the new images are copied directly from their respective static image metadata. The video metadata is disgarded.

The code also copies all other non-Live images and videos unchanged.

### AI algorithm 1

There is a `Tensorflow` powered machine learning algorithm which to re-orients the images back to their orignal orientation, as doing it manually by looking at metadata was found to be inconistent and inaccurate. They have been trained on thousands of personal images. 

There is also an equivalent and identical `Pytorch` algorithm which is used as backup if the `Tensorflow` algorithm fails (eg Tensorflow isn't installed).

I have left the algorithm codes here if you want to train them on your own photos. This code automatically changes the orientation of the images for training, so all you need to do it provide the PATH to the images folder. 
 - I have also left a code that you should run on your images folder to reduce the image size so the algorithm can train on these smaller images, otherwise it's too computationally expensive.

AI algorithm stats:
 - accuracy: 0.9996
 - loss: 9.9206e-04
 - val_accuracy: 1.0000
 - val_loss: 0.0012


### AI algorithm 2




## Apple Format Help

**Era**	                                **Image**	    **Video**
2015–2017	                              .jpg	          .mov
2017–present (High Efficiency)	          .heic	          .mov
2017–present (Most Compatible)	          .jpg	          .mp4

- These are the only format-types that are considered in this program. `.heic + .mp4` is never valid. 
- Photos before Sept 2015 will not be considered (Live Photos introduced Sept 2015), and will simply be added to the output folder __unchanged__.