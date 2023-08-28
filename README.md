# WGAN-GP_WG_nanophotonics

NOTICE: All files havent been uploaded, Access available when published. 

This code is for PhD project of inverse design of nanophotonics structure. Open Access Available once Published.

Platform: 

CPU: AMD R9 5900x 4.6Ghz

GPU: ASUS ROG strix 3090oc 

RAM: Crucial 32GB 4200Mhz oc

Pytorch Version: 1.11

Conda Version: 4.14.0


In this project, we demoasted the random waveguide design in image size 64 x 64, physical size 5 x 5 um. 
The waveguide was stimulated via the FDTD method using custom Fortran code. 
Traditional inverse-design waveguide are using all kind of optimization techniques, such as PSO, GA.
However, these methods are all time-comsuption heavy. 

Therefore, Deep Learning is introduced. A generative adversarial network (GAN) is used for this specific project.

The problem with the inverse design of the random geometries waveguide is the whole design space: 
Large Design space but with a random number of holes

There are no patten or guidance for Neural networks to learn. 

Thus, regular DNN or CNN are hard to train for this case. 


We chose GAN for our base network to learn the Transmission Spectra and Geometries relations. 


Conditional Deep Convolutional GAN  & Wasserstein GAN with gradient penalty are implanted. 



![Unseen_spectra_new_gemo](https://github.com/ZooBeasts/WGAN-GP-Inverse-Design-Waveguide-nanophotonics/assets/75404784/9d707031-0185-4f17-bdd7-0fe1147f63c6)








//////////////////////////////////////////////////////////////

How to use:


Dataloader.py contains the one2one correlation between transmission spectra and real geometrical images.


gradientpenalty.py contains the gradient penalty calculation and definition. 


main.py contains training step and run the code. 


Model.py contains the main model for WGAN-gp


Generator.py contains loading saved training info and generate new images.
















