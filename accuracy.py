import create_dataset

test_inputs = create_dataset.get_test_inputs()
test_labels = create_dataset.get_test_labels()

def acc(model):

        pred_labels = model(test_inputs).argmax(dim=1)

        acc = (pred_labels == test_labels).float().mean().item()

        print("\nAccuracy = ")
        print(acc * 100)
        return acc