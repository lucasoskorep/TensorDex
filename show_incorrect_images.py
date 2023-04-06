import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

df = pd.read_csv("sub1_non_transfer.csv")
df2 = pd.read_csv("poke_evos.csv")

evos = []

for index, row in df2.iterrows():
    print(row)
    s = ""
    s+=row["stage1"] if not pd.isnull(row["stage1"]) else ""
    s+=row["stage2"] if not pd.isnull(row["stage2"]) else ""
    s+=row["stage3"] if not pd.isnull(row["stage3"]) else ""
    evos.append(s.lower().replace(" ", "-").rstrip())


incorrect = df[df["prediction"]!= df["true_val"]]

total_same_fam = 0
for index, row in incorrect.iterrows():
    img = mpimg.imread("./SingleImageTestSet/" + row['fname'])
    imgplot = plt.imshow(img)
    title = f"Predicted - {row['prediction']}, Actual - {row['true_val']}"
    for evo in evos:
        if row['prediction'] in evo and row['true_val'] in evo:
            title+=f"\n same family name detected - {evo}"
            total_same_fam+=1
    plt.title(title)
    plt.show()


print(f"The total number of incorrect entries from same families is {total_same_fam} - {total_same_fam/len(incorrect)}")