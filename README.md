#  Supervised SNN Based on Gradient Descent in TensorFlow (v1)
-----------------------
*This repo attempts to proposes a supervised learning algorithm of SNN by using spike sequences with complex spatio-temporal information. We explore an error back-propagation method of SNN based on gradient descent. The chain rule proved mathematically that it is sufficient to update the SNN’s synaptic weights by directly using an optimizer.  Utilizing the TensorFlow framework, a bilayer supervised learning SNN is constructed from scratch. We take the lead in the application of SAR image classification and conduct experiments on the MSTAR dataset.*
 
## Requirements
- GPU MEMORY >= 8GB
- tensorflow >= 1.1.4
- imageio >= 2.6.1
- numpy

## Dataset Preparation
To accelerate and optimize the algorithm implementation, we directly input spike sequences to the SNN. Those spike sequences (correspond to the input images) are obtained through SNN’s receptive field and spike generator. The main reference is from [https://github.com/Shikhargupta/Spiking-Neural-Network](https://github.com/Shikhargupta/Spiking-Neural-Network) The spike sequences of BMP-2, BTR-60, and T-72 are as follows (example for one of the dataset samples):
<p align="center">
  <img src=".\fig\input1.png" width=280 height=260>
  <img src=".\fig\input2.png" width=280 height=260>
  <img src=".\fig\input3.png" width=280 height=260>
</p>

The guidance signal is the convergence membrane potential of output layer neurons based on an unsupervised learning method: [https://www.preprints.org/manuscript/202102.0083/v1](https://www.preprints.org/manuscript/202102.0083/v1 "Unsupervised Learning Method for SAR Image Classification Based on Spiking Neural Network"). They are features that neurons abstract from each category of the input images. The membrane potential guidance signals of BMP-2, BTR-60, and T-72 are as follows:
<p align="center">
  <img src=".\fig\guild1.png" width=1037 height=419>
  <img src=".\fig\guild2.png" width=1034 height=419>
  <img src=".\fig\guild3.png" width=1037 height=419>
</p>

**Note: In this first version, we only open source three samples (one image for each category) of the dataset, which are used to run the code. But a dataset with only three samples will seriously affect the classification accuracy. Before executing the project, please expand the sample number to a certain value (dataset scale) or replace them with your own dataset. When creating your own data set, please save the spike sequences as `.pkl` files.**

## Training
For training, you need to configure the image loading path `TRAIN_DIR`, `TEST_DIR` and the model event output path `TRAIN_WRITER_DIR`, `TEST_WRITER_DIR` and `CHECKPOINT_FL`. Then run `learning.py`.

You can modify the value of `SAVE_INTERVAL` to control the saving interval of the model. Normally, we recommend that the values of `SAVE_INTERVAL` and `TRAIN_INTERVAL` are equal.

**Note: `BATCH_SIZE = 1` is unchangeable for online learning.**

## Test
In each saving interval of the model, "test" runs automatically after "training". The current model's accuracy displays on the command-line interface.

Classification accuracy versus training epoch of the supervised learning bilayer SNN:
<p align="center">
  <img src=".\fig\snnacc.png" width=687 height=522>
</p>

where the purple shade represents the total performance range of the SNN under multiple repeated experiments.
## Curves and Feature Visualization on Tensorboard
We can obtain loss curves and feature maps of the SNN's synapse weights via **Tensorboard**.

The curve of Huber Loss versus training epoch of the supervised learning bilayer SNN is as follows:
<p align="center">
  <img src=".\fig\huberloss.png" width=1164 height=366>
</p>

Feature visualization via Tensorboard (and output heat maps by Matlab):
<p align="center">
  <img src=".\fig\snnvis.png" width=1037 height=903>
</p>

The above figures are the visualization of synaptic weights of the supervised learning SNN. Each row from top to bottom is BMP-2, BTR-60, and T-72, respectively, and each column from left to right is the feature map via 5, 10, and 25 training epochs, respectively.

Tensor graph of the proposed SNN is:
<p align="center">
  <img src=".\fig\tensorgraph1.png" width=1170 height=501>
  <img src=".\fig\tensorgraph2.png" width=795 height=203>
  <img src=".\fig\tensorgraph3.png" width=1269 height=258>
</p>

## Hyperparameters
The performance of SNN is sensitive to its hyperparameters. Even for the bilayer SNN, the hyperparameters need to be debugged according to the LIF model‘s biological characteristics and the data set.
<p align="center">
  <img src=".\fig\hyperpara.png" width=908 height=526>
</p>


## Engineering Operation
Our experimental platform is Ubuntu 18.04, 64G memory, and GPU Quadro RTX8000. When the input image size is 128×128, GPU’s occupation stables at 62%, and with the fps≈7.

## TODO:
Pull requests are welcome.

- [x] Directly input in the form of an image
- [ ] Build deep SNN
- [ ] Improve SNN's model training speed (optimize tensor calculation process in TensorFlow or use PyTorch)
- [ ] Offline learning (i.e., batchsize >1)
- [ ] Multi-GPU support
- [ ] Model pretrained on MSTAR

------
> Author: Jiankun Chen
> 
> This version was updated on 4/17/2021 3:52:56 AM 
