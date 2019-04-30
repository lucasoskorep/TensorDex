import pandas as pd
import os

pd.DataFrame(sorted([f.name for f in os.scandir("./data/train") if f.is_dir()])).to_csv("labels.txt", index=False, header=False)
