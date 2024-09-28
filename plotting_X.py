import matplotlib.pyplot as plt
import pca
import create_dataset

labels_colors = {1 : "b",
                 2: "g",
                 3 : "r",
                 4 : "c",
                 5: "m",
                 6: "y",
                 7: "k",
                 8: "orange",
                 9: "purple",
                 0: "pink"
                 }

def plot(X, y, epoch):
    X_pca = pca.pca_of_two(X)
    y_np = y.numpy()

    pc1 = X_pca[:, 0]
    pc2 = X_pca[:, 1]


    for i in range(y.shape[0]):
        
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = labels_colors[y_np[i]]

        plt.scatter(x_axis, y_axis, color=color, s=1)

    plt.title('X sample scatter plot during training '  + str(epoch))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)

    plt.savefig("X during training scatter plot for sample " + str(epoch))
    plt.show()





