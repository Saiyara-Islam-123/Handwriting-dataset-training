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

if __name__ == '__main__':
    X = create_dataset.get_inputs()
    y_np = create_dataset.get_labels().numpy()
    X_pca = pca.pca_of_two(X)

    pc1 = X_pca[:, 0]
    pc2 = X_pca[:, 1]


    for i in range(2000):
        
        x = pc1[i]
        y = pc2[i]

        color = labels_colors[y_np[i]]

        plt.scatter(x, y, color=color, s=1)

    plt.title('X sample scatter plot pre-training')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)

    plt.savefig("X pre-training scatter plot for sample")
    plt.show()






