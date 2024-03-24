import os
from PIL import Image
import numpy as np
import torch

# Set the input and output directories
input_dir = ''
output_dir = ''

# Create the output directory if it doesn't already exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Open the image file using PIL
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)
    img_gray = img.convert('L')

    # Convert the image to a NumPy array
    img_np = np.array(img_gray)

    # Convert the NumPy array to a PyTorch tensor
    img_tensor = torch.from_numpy(img_np)

    # Replace values below 200 with 0, and set all other values to 1
    img_tensor[img_tensor <= 80] = 3
    img_tensor[img_tensor > 80] = 1

    # Reshape the tensor from 64 x 64 to 4096 x 1
    # img_tensor = img_tensor.reshape((64, 64))
    img_tensor = img_tensor.t()
    img_tensor = img_tensor.reshape((4096,1))
    img = torch.flatten(img_tensor, 0, -1)

    # Save the altered tensor as a text file
    tensor_name = os.path.splitext(filename)[0] + '.txt'
    tensor_path = os.path.join(output_dir, tensor_name)
    np.savetxt(tensor_path, img_tensor.numpy(), fmt='%d')
