import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/electricsheepafrica/nigerian_transport_and_logistics_fuel_consumption/nigerian_transport_and_logistics_fuel_consumption.parquet")
print(df)

df.to_csv('nigerian_transport_and_logistics_fuel_consumption.csv', index=False)