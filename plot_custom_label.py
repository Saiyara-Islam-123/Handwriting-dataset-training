import matplotlib.pyplot as plt
import numpy as np

def custom_label(x_values, y_values, cat, color):

    for i in range(len(x_values)):
        x = x_values[i]
        y = y_values[i]
        linestyle_text = cat
        plt.text(x, y, linestyle_text, fontsize=10, color=color, ha='center', va='center')