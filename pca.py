from sklearn.decomposition import PCA


def pca_of_two(matrix):
    pca_original = PCA(n_components=2)
    return pca_original.fit_transform(matrix)
