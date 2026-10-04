from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
import yaml

with open("params.yaml", "r") as f:
    p = yaml.safe_load(f)["train"]

data = Path("data/processed")
models = Path("models")
models.mkdir(parents=True, exist_ok=True)

x_train = np.load(data / "x_train.npy")
y_train = np.load(data / "y_train.npy")
x_val = np.load(data / "x_val.npy")
y_val = np.load(data / "y_val.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(p["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(p["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=p["epochs"],
    batch_size=p["batch_size"],
    verbose=2,
)

model.save(models / "model.h5")

keys = list(history.history.keys())
with open(models / "history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + keys)
    for i in range(len(history.history[keys[0]])):
        writer.writerow([i + 1] + [history.history[k][i] for k in keys])

print("Model and history saved to models/")