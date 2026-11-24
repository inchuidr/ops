from mlflowops.sample_data import generate

generate(1500, 42).to_csv("data/sample_churn.csv", index=False)
print("wrote data/sample_churn.csv")
