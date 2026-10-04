from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["preprocess"]

raw = Path("data/raw")
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

NORMALIZATION_DIVISOR = 255.0
x_train = np.load(raw / "x_train.npy").astype("float32") / NORMALIZATION_DIVISOR
y_train = np.load(raw / "y_train.npy")
x_test = np.load(raw / "x_test.npy").astype("float32") / NORMALIZATION_DIVISOR
y_test = np.load(raw / "y_test.npy")
x_train = np.clip(x_train, 0.0, 1.0)
x_test = np.clip(x_test, 0.0, 1.0)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=params["validation_size"],
    random_state=params["seed"],
    stratify=y_train,
)

np.save(out / "x_train.npy", x_train)
np.save(out / "y_train.npy", y_train)
np.save(out / "x_val.npy", x_val)
np.save(out / "y_val.npy", y_val)
np.save(out / "x_test.npy", x_test)
np.save(out / "y_test.npy", y_test)

print("Processed data saved to data/processed/")