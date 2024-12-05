import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

def load_mnist_data(batch_size=32):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return dataset


def get_category_data(dataset, category):
    """Retrieve all images in the dataset belonging to a specified category and flatten them."""
    data = [img.view(-1).numpy() for img, label in dataset if label == category]
    return torch.tensor(data)


def calculate_average_cosine_similarity(data1, data2):
    """Calculate the average cosine similarity between two sets of flattened images."""
    similarity = cosine_similarity(data1, data2)
    upper_triangle = similarity[np.triu_indices_from(similarity, k=1)]
    return upper_triangle.mean()


def compute_pairwise_similarities(dataset):
    categories = range(10)
    avg_similarities = np.zeros((10, 10))

    for cat1, cat2 in itertools.combinations(categories, 2):
        data1 = get_category_data(dataset, cat1)
        data2 = get_category_data(dataset, cat2)
        print(data1.shape)
        print(data2.shape)

        # Calculate the average cosine similarity for the pair (cat1, cat2)
        avg_similarity = calculate_average_cosine_similarity(data1, data2)
        avg_similarities[cat1, cat2] = avg_similarity
        avg_similarities[cat2, cat1] = avg_similarity  # Make matrix symmetric
        print(f"Average Cosine Similarity for {cat1} vs {cat2}: {avg_similarity}")

    return avg_similarities


def plot_similarity_matrix(matrix):

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("white_black", ["white", "black"])

    # Extract the upper triangular part of the matrix, excluding the diagonal
    upper_triangular_values = matrix[np.triu_indices_from(matrix, k=1)]

    # Get the min and max values from the upper triangular part
    vmin = upper_triangular_values.min()
    vmax = upper_triangular_values.max()
    plt.figure(figsize=(8, 6))
    # Create a mask for the lower triangular part and the diagonal
    mask = np.tril(np.ones_like(matrix, dtype=bool))
    # Apply the mask to the matrix before plotting
    masked_matrix = np.ma.array(matrix, mask=mask)
    plt.imshow(masked_matrix, cmap=custom_cmap ,vmin=vmin, vmax=vmax)
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(ticks=range(10), labels=range(10))
    plt.yticks(ticks=range(10), labels=range(10))
    plt.title("Pairwise Cosine Similarity Matrix")
    plt.xlabel("Digit Category")
    plt.ylabel("Digit Category")

    #plt.savefig("Cosine Similarity Pre-training" + ".png")
    plt.show()


if __name__ == "__main__":
    # Ensure the plots directory exists
    os.makedirs('./plots', exist_ok=True)
    similarity_matrix_path = './plots/similarity_matrix.npy'


        # Load MNIST dataset
    dataset = load_mnist_data()

        # Compute pairwise cosine similarities
    pairwise_similarity_matrix = compute_pairwise_similarities(dataset)

        # Save the similarity matrix
    np.save(similarity_matrix_path, pairwise_similarity_matrix)
    print(f"Saved similarity matrix to {similarity_matrix_path}")

    # Plot the similarity matrix
    plot_similarity_matrix(pairwise_similarity_matrix)
