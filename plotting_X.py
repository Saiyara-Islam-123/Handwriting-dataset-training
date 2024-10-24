import matplotlib.pyplot as plt
import tnse
import create_dataset
import torch

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

def filter(X, y, list_digits):
    X_filtered = []
    y_filtered = []

    for i in range(5000):

        if y[i] in list_digits:
            y_filtered.append(y[i])
            X_filtered.append(X[i])


    return torch.stack(X_filtered), torch.stack(y_filtered)




def plot(X, y, epoch, is_sup):
    X_tnse = tnse.of_two(X.view(X.shape[0], -1).detach().numpy())
    y_np = y.numpy()

    pc1 = X_tnse[:, 0]
    pc2 = X_tnse[:, 1]


    for i in range(y.shape[0]):
        
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = labels_colors[y_np[i]]

        plt.scatter(x_axis, y_axis, color=color, s=1)

    plt.title('X sample scatter plot during ' + is_sup + ' training '  + str(epoch))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)

    plt.savefig(is_sup + " X during training scatter plot for sample " + str(epoch))
    plt.show()


if __name__ == '__main__':
    X,y = (filter(create_dataset.get_inputs(), create_dataset.get_labels(), [0, 1, 4,9]))

    plot(X.reshape(X.shape[0], 784), y, -1, "Pre-training")
