import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

df = pd.read_csv("4 9, batch of 50, plotting for every batch, between unsup and sup.csv")

df.drop(0, axis=0, inplace=True)

x = []
for i in range(len(df["within_4"])):
    x.append(i)
    training_phase = df["training_phase"].iloc[i]
    print(i, training_phase)


#plt.plot(x, df["within_4"], color="green", label="within_4", alpha=0.7)
#plt.plot(x, df["within_9"], color="lime", label="within_9", alpha=0.7)
#plt.plot(x, df["between_4_9"], color="blue", label="between_4_9", alpha=0.5)
#plt.axvline(x=119, color='r', linestyle='--', linewidth=1)


sb.regplot(x=x, y=df["within_4"], color="green", label="within_4", lowess=True, scatter=True, scatter_kws={'alpha' : 0.2}, line_kws={'alpha': 1})
sb.regplot(x=x, y=df["within_9"], color="lime", label="within_9", lowess=True, scatter=True, scatter_kws={'alpha' : 0.2}, line_kws={'alpha': 1})
sb.regplot(x=x, y=df["between_4_9"], color="blue", label="between_4_9", lowess=True, scatter=True, scatter_kws={'alpha' : 0.2}, line_kws={'alpha': 1})
plt.axvline(x=119, color='r', linestyle='--', linewidth=1)



plt.xlabel("Phases of training")
plt.ylabel("Distances")
plt.title("4,9")
plt.legend()
plt.show()