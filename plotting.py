import matplotlib.pyplot as plt
import tnse
import create_dataset
import torch
torch.random.manual_seed(0)
import neural_networks

labels_colors = {1 : "limegreen",
                 2: "pink",
                 3 : "r",
                 4 : "c",
                 5: "m",
                 6: "y",
                 7: "k",
                 8: "orange",
                 9: "purple",
                 0: "green"
                 }

def filter(X, y, list_digits):
    X_filtered = []
    y_filtered = []

    for i in range(y.shape[0]):

        if y[i] in list_digits:
            y_filtered.append(y[i])
            X_filtered.append(X[i])


    return torch.stack(X_filtered), torch.stack(y_filtered)


def plot(X, y, epoch, batch, is_sup, name, loc):
    X_tnse = tnse.of_two(X.view(X.shape[0], -1).detach().numpy())
    y_np = y.numpy()

    pc1 = X_tnse[:, 0]
    pc2 = X_tnse[:, 1]


    for i in range(y.shape[0]):
        
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = labels_colors[y_np[i]]

        plt.scatter(x_axis, y_axis, color=color, s=10)
    plt.scatter([], [], color="limegreen", label="1")
    plt.scatter([], [], color="green", label='0')

    plt.title('Scatter plot during ' + is_sup + ' training, epoch: '  + str(epoch) + " batch: " + str(batch))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()

    plt.savefig(f"{loc}/"+name.strip(".pth")+".png")

    plt.show()

def plot_across_batches(batch, weights,loc):
    X, y = (filter(create_dataset.get_inputs(), create_dataset.get_labels(), [0, 1]))

    if "unsup_model.pth" in weights.split(" "):
        is_sup = "unsup"
    else:
        is_sup = "sup"

    indices = torch.randperm(X.size(0))
    split = int(0.03 * X.size(0))

    indices = indices[:split]
    X, y = X[indices], y[indices]

    if is_sup=="sup":
        unsup_model = neural_networks.AutoEncoder()
        unsup_model.load_state_dict(torch.load("unsup_weights/599 100 0.05 unsup_model.pth"))
        sup_model = neural_networks.LastLayer(unsup_model)
        sup_model.load_state_dict(torch.load("sup_weights/0.005/"+weights))
        _ = sup_model(X)
        encoder_outputs = sup_model.autoencoder_output


    else:
        unsup_model = neural_networks.AutoEncoder()
        unsup_model.load_state_dict(torch.load("unsup_weights/"+weights))
        _ = unsup_model(X)
        encoder_outputs = unsup_model.encoded

    plot(encoder_outputs, y, epoch=0, batch=batch, is_sup=is_sup, name=weights.split(" ")[0], loc=loc)

if __name__ == '__main__':
    '''
    X, y = (filter(create_dataset.get_inputs(), create_dataset.get_labels(), [0, 1]))
    indices = torch.randperm(X.size(0))
    split = int(0.03 * X.size(0))

    indices = indices[:split]
    X, y = X[indices], y[indices]

    plot(X, y, epoch="", batch="", is_sup="", name="no_train", loc="new_plots/scatter_plots/no_train")


    '''
    for i in range(600):
            if i % 10 == 0:
                plot_across_batches(weights=f"{int(i)} 100 0.05 unsup_model.pth", loc="new_plots/scatter_plots/unsup", batch=(i))
    
