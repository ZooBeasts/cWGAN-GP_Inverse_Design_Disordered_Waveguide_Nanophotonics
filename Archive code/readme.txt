#here is the readme


How to use:
This code was written and run in Windows 10. Therefore, number_workers = 0 If you want to run it in Linux, please remember to change the number_workers to your desired number.

Steps:

if your data is split in *. txt, you can use
python convert_array2fig.py
Notice: Remember to change color, size, row and col size to fit your data.

Check Dataloder.py
python Dataloder.py
Notice: Check your data path, image path, data index to avoid error

Check Model.py
python Model.py
Notice: Model.py contains some hyperparameters ( which are also can be controlled in main.py)

Check main.py and run
python main.py
Notice: main.py contains hyperparameters and need to be changed to suit your data. Dont forgot the save location.

Check Unseenprediction and seenprediction.py
python Unseenprediction.py
python seenprediction.py
Notice: path location and Z_dim size to avoid errors.

Load generated geometry, go through Norm_image.py to remove noise
python Norm_image.py
Notice: check >, <, = value to estimate noise from images

Do regression in Regression folder
python Nonlinear_CNN.py
Notice: remember to change hyperparameters to fit your data, repeat 5 times for different power input

Prediction using Nonlinear_pred1e0.py, etc
python Nonlinear_pred1e0.py
Notice: Where to predict the output, etc

Now you have the results. and Have Fun with it.
