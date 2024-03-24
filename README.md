# Generative model for inverse design and forward predictions of disordered waveguides in the linear and nonlinear regimes

## COMING SOON (Updated readme.*), dont Panic if you see old code disappeared, they are in archive code folder. 

### Ziheng Guo<sup>1</sup>, Zhongliang Guo<sup>2</sup>, Oggie Arandelovic<sup>2</sup>, Andrea di Falco<sup>1*</sup>
###  <sup>1</sup> School of Physics and Astronomy, University of St. Andrews, Fife, KY16 9SS, United Kingdom
###  <sup>2</sup> School of Computer Science, University of St. Andrews, Fife, KY16 9SS, United Kingdom

### 22/02/2024 Work will be presented in Machine Learning in Photonics, SPIE PHOTONICS EUROPE 8 April 2024, Strasbourg, France. 
![inversedeignnew](https://github.com/ZooBeasts/cWGAN-GP_Inverse_Design_Disordered_Waveguide_Nanophotonics/assets/75404784/03acddf2-3532-4f4d-bee4-8055ecdd47d4)

![forward-p](https://github.com/ZooBeasts/cWGAN-GP_Inverse_Design_Disordered_Waveguide_Nanophotonics/assets/75404784/e1c9b779-acef-43f2-a546-743612ed7edb)


Platform: 

CPU: AMD R9 5900x 4.6Ghz

GPU: ASUS ROG strix 3090oc 

RAM: Crucial 32GB 4200Mhz oc

Pytorch Version: 2.0.1 + cuda 11.7

Conda Version: v23.7.2


In this project, we demonstrated the random waveguide design in image size 64 x 64 x 1 channel, and physical size 6.4 x 6.4 um. 
The waveguide was stimulated via the FDTD method using custom Fortran code. Ref: Optical parametric oscillations in isotropic photonic crystals, Claudio Conti, Andrea Di Falco, and Gaetano Assanto. 
https://opg.optica.org/oe/fulltext.cfm?uri=oe-12-5-823&id=79198

Traditional inverse design waveguides use all kinds of optimization techniques, such as PSO, GA. 
However, these methods are all very time-consuming.  

Therefore, Data-driven Machine learning is introduced. A Wasserstein generative adversarial network (WGAN-gp) is used for inverse design random waveguide, and CNN is used for forward prediction.
## GAN:
### Input: Reference Linear transmission spectra data
### Output: Generated new geometries. 
### Aim to solve three different inverse design problems:
![image](https://github.com/ZooBeasts/cWGAN-GP_Inverse_Design_Disordered_Waveguide_Nanophotonics/assets/75404784/726d58bf-23fe-4ed8-a6eb-086730270e86)




## Forward Prediction:
### Input: Geometries ( from GAN or FDTD )
### Output: Transmission Spectra (reference and nonlinear)

The model prediction is measured with MSE, and the acceptable error is within 10%. Due to CNN kernel scanning ability for random, sparse type doesn't perform well.

## Our work considered more randomness in order to extend the overall degree of freedom of waveguide design space. Therefore, it is possible to improve model generalization to avoid mode collapse during the training, as our physical data is sparse type. 200 etched holes in total 4096 design space.


//////////////////////////////////////////////////////////////////////////////////

# How to use: 

This code was written and run in Windows 10. Therefore, number_workers = 0 
If you want to run it in Linux, please remember to change the number_workers to your desired number. 

Steps: 

1. if your data is split in *. txt, you can use 
#### python convert_array2fig.py
Notice: Remember to change color, size, row and col size to fit your data.

2. Check Dataloder.py
#### python Dataloder.py
Notice: Check your data path, image path, data index to avoid error

3. Check Model.py in Model folder
Notice: Model.py contains some hyperparameters (where some are also can be controlled in main.py)

4. Check main.py and run
Notice: main.py contains hyperparameters and need to be changed to suit your data. Dont forgot the save location. 

5. Check Pre_Known.py and Unknown.py
Notice: path location and Z_dim size to avoid errors.

6. Load generated geometry, go through Norm_image.py to remove noise
#### python Norm_image.py
Notice: check >, <, = value to estimate noise from images

7. Do regression in Nonlinear folder
#### python Nonlinear_CNN.py
Notice: remember to change hyperparameters to fit your data, repeat 10 times for different power input

8. Prediction using Nonlinear_pred1e0.py, etc
#### python Nonlinear_pred1e0.py 
Notice: Where to predict the output, etc



















