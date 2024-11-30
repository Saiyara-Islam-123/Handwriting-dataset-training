import pandas as pd
import numpy as np

def save(d_outputs, d_labels, epoch, is_sup):
    df = pd.DataFrame()
    print("hey")

    df["outputs"] = np.array(d_outputs)
    df["labels"] = np.array(d_labels)
    df.to_csv(is_sup + " " + " epoch: " + str(epoch) +'.csv', index=False)