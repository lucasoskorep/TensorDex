import pandas as pd
df = pd.read_csv("models/keras/pt-fl-fbn-efficientnet_v2s-d1024-do0.5-l11.e-04-l21.e-04-8105-second_stage.csv")

print(df.loc[df["prediction"] != df["true_val"]])
