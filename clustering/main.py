from sklearn.cluster import KMeans
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "data", "colors.csv")
df = pd.read_csv(csv_path)

X = df[["red", "green", "blue"]].values
X_scaled = X / 255.0
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X_scaled)
df["label"] = kmeans.labels_

label_path = os.path.join(BASE_DIR, "..", "data", "colors_labeled.csv")
df.to_csv(label_path, index=False)
print("Done labeling!")