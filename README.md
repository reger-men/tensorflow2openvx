# Convert tensorflow pre-trained models to opnevx Graph
This converter was created to convert my own (in Keras with tensorflow as backend trained) Model to Openvx Graph.
The reason being that a direct comparison of the performance between openvx on AMD Hardware and TF_inference on Nvidia GPUs.

This implementation is still not fully implemented. However much I tried to create a expandable structure to make it easy to expand the source code.

Since a [caffe converter](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/blob/master/utils/inference_generator/src/caffe2openvx.cpp) already exists, I tried to keep the structure similar for easy integration into [amdovx-modules](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/tree/master).

## dependencies ##
To get run this project you have to download and build [tensorflow](https://www.tensorflow.org/install/install_sources). This should be fixed by using [NNEF](https://www.khronos.org/nnef)

## dependencies ##
MIT License
(c) reger-men 
