import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

dataset = pd.read_csv('.csv',header=None)


num_columns = dataset.shape[1]
submatrix_width = 5
num_submatrices = num_columns // submatrix_width

for i in range(num_submatrices):
    start_col = i * submatrix_width
    end_col = start_col + submatrix_width

    submatrix = dataset[:, start_col:end_col]

    np.savetxt(f'submatrix_{i + 1}.txt', submatrix, delimiter='\t', fmt='%.6f')  # Save as .txt

    print(f'Saved submatrix_{i + 1}.txt')



# fig, ax1 = plt.subplots()
# c = ax1.pcolor(dataset1,cmap='viridis')
# fig.tight_layout()
# plt.axis('off')
# plt.show()