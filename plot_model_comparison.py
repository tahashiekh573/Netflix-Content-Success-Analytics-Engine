
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("model_comparison.csv")
df = df.sort_values("weighted_f1", ascending=True)

plt.figure(figsize=(9, 5))
plt.barh(df["model"], df["weighted_f1"])
plt.xlim(0, 1.05)
plt.xlabel("Weighted F1 Score")
plt.title("Netflix Content Type Classification: Model Comparison")
plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=150)
plt.close()

print("Saved: reports/model_comparison.png")
