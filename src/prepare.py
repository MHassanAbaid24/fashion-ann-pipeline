from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

np.save(OUT / "x_train.npy", x_train)
np.save(OUT / "y_train.npy", y_train)
np.save(OUT / "x_test.npy", x_test)
np.save(OUT / "y_test.npy", y_test)

print("Raw Fashion-MNIST saved to data/raw/")