import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = tf.keras.models.load_model("models/model.h5")
data = Path("data/processed")

x_test = np.load(data / "x_test.npy")
y_test = np.load(data / "y_test.npy")

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(accuracy)}, f, indent=2)

print(f"Test accuracy: {accuracy:.4f}")