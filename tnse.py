from sklearn.manifold import TSNE


def of_two(matrix):
    tsne = TSNE(n_components=2)
    return tsne.fit_transform(matrix)
