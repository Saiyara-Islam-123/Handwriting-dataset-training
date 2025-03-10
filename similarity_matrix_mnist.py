#Professor Christian's code

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors



def load_mnist_data(batch_size=1):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return dataset


def get_category_data(dataset_img, dataset_label, category):
    """Retrieve all images in the dataset belonging to a specified category and flatten them."""


    data = []

    for i in range(len(dataset_img)):
        img = dataset_img[i]
        label = dataset_label[i]

        if label == category:
            data.append(img)


    return torch.tensor(np.array(data))


def calculate_average_cosine_similarity(data1, data2):
    """Calculate the average cosine similarity between two sets of flattened images."""




    if (len(data1.shape)) == 2:



        data1 = data1.reshape(data1.shape[0], 128)
        data2 = data2.reshape(data2.shape[0], 128)

    else:
        data1 = data1.reshape(data1.shape[0], 784)
        data2 = data2.reshape(data2.shape[0], 784)

    dissimilarity = 1 - cosine_similarity(data1, data2)
    upper_triangle = dissimilarity[np.triu_indices_from(dissimilarity, k=1)]
    return upper_triangle.mean()


def compute_pairwise_similarities(dataset_img, dataset_label):
    categories = range(10)
    avg_similarities = np.zeros((10, 10))

    for cat1, cat2 in itertools.product(categories, repeat=2):
        data1 = get_category_data(dataset_img, dataset_label, cat1)
        data2 = get_category_data(dataset_img,dataset_label, cat2)

        # Calculate the average cosine similarity for the pair (cat1, cat2)
        avg_similarity = calculate_average_cosine_similarity(data1, data2)
        avg_similarities[cat1, cat2] = avg_similarity
        avg_similarities[cat2, cat1] = avg_similarity  # Make matrix symmetric
        print(f"Average Distance for {cat1} vs {cat2}: {avg_similarity}")

    return avg_similarities

def upper_triangle (matrix):
    rows, cols = matrix.shape
    upper_triangular_values = []

    for i in range(rows):
        for j in range(cols):
            if j >= i:
                upper_triangular_values.append(matrix[i, j])

    return upper_triangular_values

def plot_similarity_matrix(matrix, s):

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("pink_black", ["pink", "black"])

    # Extract upper triangular values, including the diagonal
    upper_triangular_values = upper_triangle(matrix)

    # Get the min and max values from the upper triangular part
    vmin = min(upper_triangular_values)
    vmax = max(upper_triangular_values)
    plt.figure(figsize=(8, 6))
    # Create a mask for the lower triangular part and the diagonal
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    # Apply the mask to the matrix before plotting
    masked_matrix = np.ma.array(matrix, mask=~mask)
    plt.imshow(masked_matrix, cmap=custom_cmap ,vmin=vmin, vmax=vmax)
    plt.colorbar(label='Distance')
    plt.xticks(ticks=range(10), labels=range(10))
    plt.yticks(ticks=range(10), labels=range(10))
    plt.title("Pairwise Distance Matrix " + str(s))
    plt.xlabel("Digit Category")
    plt.ylabel("Digit Category")
    plt.savefig("Cosine Similarity " +str(s)+".png")
    plt.show()


'''
if __name__ == "__main__":
    # Ensure the plots directory exists
    os.makedirs('./plots', exist_ok=True)
    similarity_matrix_path = './plots/similarity_matrix.npy'

    if os.path.exists(similarity_matrix_path):
        # Load the existing similarity matrix
        pairwise_similarity_matrix = np.load(similarity_matrix_path)
        print("Loaded similarity matrix from file.")
    else:
        # Load MNIST dataset
        dataset = load_mnist_data()

        # Compute pairwise cosine similarities
        pairwise_similarity_matrix = compute_pairwise_similarities(dataset)

        # Save the similarity matrix
        np.save(similarity_matrix_path, pairwise_similarity_matrix)
        print(f"Saved similarity matrix to {similarity_matrix_path}")

    # Plot the similarity matrix
    plot_similarity_matrix(pairwise_similarity_matrix)
'''