# Generative model for inverse design and forward prediction of disorder waveguide in the linear and nonlinear regime

## Finalizing, and COMING SOON (Updated readme.*)

### Ziheng Guo<sup>1</sup>, Zhongliang Guo<sup>2</sup>, Oggie Arandelovic<sup>2</sup>, Andrea di Falco<sup>1*</sup>
###  <sup>1</sup> School of Physics and Astronomy, University of St. Andrews, Fife, KY16 9SS, United Kingdom
###  <sup>2</sup> School of Computer Science, University of St. Andrews, Fife, KY16 9SS, United Kingdom

///////////////////////////////////////////////////////////////

NOTICE: All files haven't been uploaded, Access is available UPON REASONABLE REQUEST. 


This code is for Ph.D project on the inverse design of nanophotonics structure (nanopattern in waveguide).

Platform: 

CPU: AMD R9 5900x 4.6Ghz

GPU: ASUS ROG strix 3090oc 

RAM: Crucial 32GB 4200Mhz oc

Pytorch Version: 2.0.1 + cuda 11.7

Conda Version: v23.7.2


In this project, we demonstrated the random waveguide design in image size 64 x 64 x 1 channel, and physical size 5.6 x 5.6 um. 
The waveguide was stimulated via the FDTD method using custom Fortran code. Ref: Optical parametric oscillations in isotropic photonic crystals, Claudio Conti, Andrea Di Falco, and Gaetano Assanto. 
https://opg.optica.org/oe/fulltext.cfm?uri=oe-12-5-823&id=79198

Traditional inverse design waveguides use all kinds of optimization techniques, such as PSO, GA. 
However, these methods are all very time-consuming.  

Therefore, Data-driven Machine learning is introduced. A Wasserstein generative adversarial network (WGAN-gp) is used for inverse design random waveguide, and CNN is used for forward prediction.
## GAN:
### Input: Reference Linear transmission spectra data
### Output: Generated new geometries. 
### Aim to solve three different inverse design problems:
![Three statement](https://github.com/ZooBeasts/WGAN-GP-Inverse-Design-Waveguide-nanophotonics/assets/75404784/0e4d410f-04b6-4ef7-b725-09e6cd0041f6)

## Forward Prediction
### Input: Geometries ( from GAN or FDTD )
### Output: Transmission Spectra (reference and nonlinear)

The model prediction is measured with MSE, and the acceptable error is within 20%. Due to CNN kernel scanning ability for random, sparse type doesn't perform well.

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

3. Check Model.py
#### python Model.py
Notice: Model.py contains some hyperparameters ( which are also can be controlled in main.py)

4. Check main.py and run
#### python main.py
Notice: main.py contains hyperparameters and need to be changed to suit your data. Dont forgot the save location. 

5. Check Unseenprediction and seenprediction.py, or Generate_new_image.py
#### python Unseenprediction.py
#### python seenprediction.py
#### python Generate_new_image.py
Notice: path location and Z_dim size to avoid errors.

6. Load generated geometry, go through Norm_image.py to remove noise
#### python Norm_image.py
Notice: check >, <, = value to estimate noise from images

7. Do regression in Regression folder
#### python Nonlinear_CNN.py
Notice: remember to change hyperparameters to fit your data, repeat 5 times for different power input

8. Prediction using Nonlinear_pred1e0.py, etc
#### python Nonlinear_pred1e0.py 
Notice: Where to predict the output, etc

Now you have the results. and Have Fun with it. 





/////////////////////////////////////////////////////////////////////////////////















