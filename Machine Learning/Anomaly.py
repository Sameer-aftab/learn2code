import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Step 1: Simulate health data
np.random.seed(42)
heart_rate = np.random.normal(75, 5, 200)  # Normal HR range
oxygen = np.random.normal(97, 1.5, 200)    # Normal O2 range

# Add a few anomalies
heart_rate[195:] = [110, 120, 125, 130, 115]
oxygen[195:] = [85, 88, 90, 87, 89]

data = pd.DataFrame({"heart_rate": heart_rate, "oxygen": oxygen})

# Step 2: Train anomaly detection model
model = IsolationForest(contamination=0.05)
model.fit(data)
data["anomaly"] = model.predict(data)

# Step 3: Visualize
normal = data[data["anomaly"] == 1]
anomaly = data[data["anomaly"] == -1]

plt.scatter(normal["heart_rate"], normal["oxygen"], label="Normal", color="blue")
plt.scatter(anomaly["heart_rate"], anomaly["oxygen"], label="Anomaly", color="red")
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Oxygen Level (%)")
plt.legend()
plt.title("Health Data Anomaly Detection")
plt.show()
