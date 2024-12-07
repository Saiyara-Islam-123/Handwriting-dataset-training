import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd



def plot_stretch_matrix(upper_triangular_values, s):
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("pink_black", ["pink", "black"])

    # Extract upper triangular values, including the diagonal
    matrix = fill_matrix(upper_triangular_values)

    # Get the min and max values from the upper triangular part
    vmin = min(upper_triangular_values.tolist())
    vmax = max(upper_triangular_values.tolist())

    

    plt.figure(figsize=(8, 6))
    # Create a mask for the lower triangular part and the diagonal
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    # Apply the mask to the matrix before plotting
    masked_matrix = np.ma.array(matrix, mask=~mask)
    plt.imshow(masked_matrix, cmap=custom_cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Distance')
    plt.xticks(ticks=range(10), labels=range(10))
    plt.yticks(ticks=range(10), labels=range(10))
    plt.title("Stretch " + str(s))
    plt.xlabel("Digit Category")
    plt.ylabel("Digit Category")
    plt.savefig("Stretch " + str(s) + ".png")
    plt.show()


def upper_triangle(csv_file_unsup, csv_file_sup):
    df_unsup = pd.read_csv(csv_file_unsup)
    df_sup = pd.read_csv(csv_file_sup)


    numpy_array_unsup = df_unsup.to_numpy()
    numpy_array_sup = df_sup.to_numpy()

    row, col = numpy_array_unsup.shape

    #print((numpy_array_sup[:, 0] - numpy_array_unsup[:, col-1]))

    return (numpy_array_sup[:, 0] - numpy_array_unsup[:, col-1])


def fill_matrix(upper_triangle):
    matrix = np.zeros((10, 10))
    index_upper_triangle = 0
    for i in range (10):
        for j in range (10):
            if j >= i:
                matrix[i, j] = upper_triangle[index_upper_triangle]
                index_upper_triangle += 1


    return matrix

if __name__ == '__main__':
    five_unsup_sup = upper_triangle("unsup_triangle 5 epochs each.csv", "sup_triangle 5 epochs each.csv")
    ten_unsup_sup = upper_triangle("unsup_triangle 10 epochs each.csv", "sup_triangle 10 epoch each.csv")

    five_pre_unsup = upper_triangle("pre-training.csv", "unsup_triangle 5 epochs each.csv")
    ten_pre_unsup = upper_triangle("pre-training.csv", "unsup_triangle 10 epochs each.csv")

    plot_stretch_matrix(five_pre_unsup, "5 epochs each, pre, unsup")
    plot_stretch_matrix(ten_pre_unsup, "10 epochs each, pre, unsup")

    plot_stretch_matrix(five_unsup_sup, "5 epochs each, unsup sup")
    plot_stretch_matrix(ten_unsup_sup, "10 epochs each, unsup sup")