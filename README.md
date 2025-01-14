# Identifying optimal metal-organic frameworks for adsorption-based carbon capture using natural language processing

Download this repository and run finetune_transformer.py to train the model on the CO2 adsorption capacity dataset. This will fine tune MOFormer on the provided CSV file in benchmark_datasets. If you want to train a different model, go into config_ft_transformer.yaml and change the dataset file path (and preferably the name) to the appropriate metadata for N2 adsorption capacity and regenerability.

- CO2 adsorption filepath: ./benchmark_datasets/hMOF/mofid/hMOF_CO2_1_small_mofid.csv
- N2 adsorption filepath: ./benchmark_datasets/hMOF/mofid/hMOF_N2_1_small_mofid.csv
- Regenerability filepath: ./benchmark_datasets/regenerability/mofid/regenerability_small_mofid.csv