# FaceMaskDetection-Rest-API

![](img/demo.png)

Original face mask detection is by [AIZOOTech](https://github.com/AIZOOTech/FaceMaskDetection)

AIZOOTECH provided ready made models to detect face masks on pictures and videos with high accuracy.

We took the pytorch image prediction and created rest-api for it using python-flask which can be used to query images from embedded devices and used for cool projects either locally or on the cloud.

* AIZOOTECH published 7959 images to train the models. The dataset is composed of [WIDER Face](http://shuoyang1213.me/WIDERFACE/) and [MAFA](http://www.escience.cn/people/geshiming/mafa.html), they verified and fixed some wrong annotations that exist in the original datasets. The datasets can be downloaded from [Google drive](https://drive.google.com/file/d/1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW/view?usp=sharing)

This is written specifically for Raspberry Pi 4 but should work with any other system as well, other systems might need to follow different way of installing the depndenicies.

## Installing virtual environment

You might want to consider to install all the packages in virtual environment just to prevent any issues that might happen during installation, to do so install virtual environment by running:

```
pip3 install virtualenv
```

Then create the virtual environment by running the following outside the cloned folder:

```
python3 -m virtualenv face_mask
```

To activate run:

```
source face_mask/bin/activate
```

## Installing dependencies

Install requirements for Pillow first:

```
sudo apt-get install libjpeg-dev zlib1g-dev
```

Install requirements for pytorch:

```
sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
```

Install requirements for numpy:

```
sudo apt install libatlas-base-dev
```

Downgrade GCC and G++ due to compability issues with Raspberry 4:

```
sudo apt-get install gcc-4.9 g++-4.9
```

Set environment variables:

```
export CC=gcc-4.9
export CXX=g++-4.9
export USE_CUDA=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_NUMPY=1
export USE_DISTRIBUTED=0
export NO_CUDA=1
export NO_DISTRIBUTED=1
export NO_MKLDNN=1
export NO_NNPACK=1
export NO_QNNPACK=1
export ONNX_ML=1
```

Run the following commands to install the dependencies:

```
pip3 install opencv-python3
pip3 install numpy
pip3 install pillow
```

Download and install pytorch:

```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
sudo -E python3 setup.py build
sudo -E python3 setup.py install
```
