from sklearn.manifold import TSNE


def of_two(matrix):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(matrix)
