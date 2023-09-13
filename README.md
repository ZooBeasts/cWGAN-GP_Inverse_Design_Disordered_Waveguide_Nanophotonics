# Generative model for inverse design and forward prediction of disorder waveguide in the linear and nonlinear regime

## Finalizing, and COMING SOON (Updated readme.*)

NOTICE: All files haven't been uploaded, Access is available UPON REASONABLE REQUEST. 


This code is for PhD project on the inverse design of nanophotonics structure (nanopattern in waveguide).

Platform: 

CPU: AMD R9 5900x 4.6Ghz

GPU: ASUS ROG strix 3090oc 

RAM: Crucial 32GB 4200Mhz oc

Pytorch Version: 1.11

Conda Version: 4.14.0


In this project, we demonstrated the random waveguide design in image size 64 x 64 x 1 channel, and physical size 5.6 x 5.6 um. 
The waveguide was stimulated via the FDTD method using custom Fortran code. Ref: Optical parametric oscillations in isotropic photonic crystals, Claudio Conti, Andrea Di Falco, and Gaetano Assanto. 
https://opg.optica.org/oe/fulltext.cfm?uri=oe-12-5-823&id=79198

Traditional inverse design waveguides use all kinds of optimization techniques, such as PSO, GA. 
However, these methods are all very time conmusing.  

Therefore, Data-driven Machine learning is introduced. A Wasserstein generative adversarial network (WGAN-gp) is used for inverse design random waveguide, and CNN is used for forward prediction.
## GAN:
### Input: Reference Linear transmission spectra data
### Output: Generated new geometry. 
### Aim to solve three different inverse design problems:
![Three statement](https://github.com/ZooBeasts/WGAN-GP-Inverse-Design-Waveguide-nanophotonics/assets/75404784/0e4d410f-04b6-4ef7-b725-09e6cd0041f6)

## Forward Prediction
### Input: Geometries ( from GAN or FDTD )
### Output: Transmission Spectra

The model prediction is measured with MSE, and the acceptable error is within 20%. Due to CNN kernel scanning ability for random, sparse types doesn't perform well.

## Our work considered more randomness in order to extend the overall degree of freedom of waveguide design space. Therefore, it is possible to improve model generalization to avoid mode collapse during the training, as our physical data is sparse type. 200 etched holes in total 4096 design space.

//////////////////////////////////////////////////////////////////////////////////

How to use: 

This code was written and run in Windows 10, therefore, number_workers = 0 
If you want to run it in Linux, please remember to change the number_workers to your desired number. 

python main.py


/////////////////////////////////////////////////////////////////////////////////















