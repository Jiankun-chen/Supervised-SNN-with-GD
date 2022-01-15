#  Supervised SNN Based on Gradient Descent in TensorFlow
-----------------------
*This repo attempts to proposes a supervised learning algorithm of SNN by using spike sequences with complex spatio-temporal information. We explore an error back-propagation method of SNN based on gradient descent. The chain rule proved mathematically that it is sufficient to update the SNN’s synaptic weights by directly using an optimizer.  Utilizing the TensorFlow framework, a bilayer supervised learning SNN is constructed from scratch. We take the lead in the application of SAR image classification and conduct experiments on the MSTAR three types of targets in SOC mode.*
 
## Requirements
- GPU MEMORY >= 8GB
- tensorflow >= 1.1.4
- imageio >= 2.6.1
- numpy

## Dataset Preparation
To accelerate and optimize the algorithm implementation, we directly input spike sequences to the SNN. Those spike sequences (correspond to the input images) are obtained through SNN’s receptive field and spike generator. The main reference is from [https://github.com/Shikhargupta/Spiking-Neural-Network](https://github.com/Shikhargupta/Spiking-Neural-Network) The spike sequences of BMP2, BTR70, and T72 are as follows (example for one of the dataset samples).

Here is an example. If we reszie the images to 16×16 size. 
Then the images of target would be:
<p align="center">
  <img src=".\github_fig\BMP2.png" width=128 height=128>
  <img src=".\github_fig\BTR70.png" width=128 height=128>
  <img src=".\github_fig\T72.png" width=128 height=128>
</p>
When the spike emission duration of a single image（SEDSI）is 70ms. And the input spikes would be:
<p align="center">
  <img src=".\github_fig\s_BMP2.png" width=600 height=975>
  <img src=".\github_fig\s_BTR70.png" width=600 height=975>
  <img src=".\github_fig\s_T72.png" width=600 height=975>
</p>

The guidance signal is the convergence membrane potential of output layer neurons based on an unsupervised learning method: [https://www.preprints.org/manuscript/202102.0083/v1](https://www.preprints.org/manuscript/202102.0083/v1 "Unsupervised Learning Method for SAR Image Classification Based on Spiking Neural Network"). They are features that neurons abstract from each category of the input images. The membrane potential guidance signals of BMP2, BTR70, and T72 are as follows:
<p align="center">
  <img src=".\github_fig\P_BMP2(sn-9566).png" width=640 height=480>
  <img src=".\github_fig\P_BTR70(sn-c71).png" width=640 height=480>
  <img src=".\github_fig\P_T72(sn-132).png" width=640 height=480>
</p>

In this first version, we only open source three samples (one image for each category) of the data set, which are used to run the code.

## Training
For training, you need to configure the image loading path `TRAIN_DIR`, `TEST_DIR` and the model event output path `TRAIN_WRITER_DIR`, `TEST_WRITER_DIR` and `CHECKPOINT_FL`. Then run `train.py`.

You can modify the value of `SAVE_INTERVAL` to control the saving interval of the model. Normally, we recommend that the values of `SAVE_INTERVAL` and `TRAIN_INTERVAL` are equal.

**Note: `BATCH_SIZE = 1` is unchangeable for online learning.**

## Test
In each saving interval of the model, "test" runs automatically after "train". 

The classification accuracy versus training epoch of the supervised learning bilayer SNN is:
<p align="center">
  <img src=".\github_fig\acc.png" width=673 height=415>
</p>

## Curves and Feature Visualization on Tensorboard
We can obtain loss curves and feature maps of the SNN's synapse weights via **Tensorboard**.

The curve of Huber Loss versus training epoch of the supervised learning bilayer SNN is as follows:
<p align="center">
  <img src=".\github_fig\huberloss.png" width=662 height=412>
</p>

Feature visualization of BMP2, BTR70, and T72 via Tensorboard is:
<p align="center">
  <img src=".\github_fig\vis_BMP2.png" width=128 height=128>
  <img src=".\github_fig\vis_BTR70.png" width=128 height=128>
  <img src=".\github_fig\vis_T72.png" width=128 height=128>
</p>

The above figures are the visualization of synaptic weights of the supervised learning SNN. Each row from top to bottom is BMP-2, BTR-60, and T-72, respectively, and each column from left to right is the feature map via 5, 10, and 25 training epochs, respectively.

The tensor graph of the proposed SNN is:
<p align="center">
  <img src=".\github_fig\net_p1.png" width=808 height=414>
  <img src=".\github_fig\net_p2.png" width=658 height=301>
  <img src=".\github_fig\net_p3.png" width=872 height=247>
</p>
## Engineering Operation
Our experimental platform is Ubuntu 18.04, 64G memory, and GPU Quadro RTX8000. When the input image size is 128×128, GPU’s occupation stables at 62%, and with the fps≈7.

## Stability
It has been found in the experiment that the input spike should be interpolated 10 times in order to prevent the training interruption caused by the instability of gradient descent. But this will cause the training speed to drop to fps≈2.8.

## TODO:
Pull requests are welcome.

- [x] Directly input in the form of an image
- [ ] Build deeper SNN
- [ ] Improve SNN's model training speed (optimize tensor calculation process in TensorFlow or use PyTorch)
- [ ] Offline learning (i.e., batchsize >1)
- [ ] Multi-GPU support
- [ ] Model pretrained on MSTAR

------
> Author: Jiankun Chen
> 
> This version was updated on 1/15/2022 6:08:25 PM  
