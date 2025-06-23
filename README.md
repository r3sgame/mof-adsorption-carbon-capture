# Identifying optimal metal-organic frameworks for adsorption-based carbon capture using natural language processing

## [Read the paper](https://docs.google.com/document/d/1zm9vep4UVej5GFiulpjX60IibpaK2EQztXNkPm2Zaeg/edit?usp=sharing)

Download this repository and run finetune_transformer.py to train the model on the CO2 adsorption capacity dataset. This will fine tune MOFormer on the provided CSV file in benchmark_datasets. If you want to train a different model, go into config_ft_transformer.yaml and change the dataset file path (and preferably the name) to the appropriate metadata for N2 adsorption capacity and regenerability.

- CO2 adsorption filepath: ./benchmark_datasets/hMOF/mofid/hMOF_CO2_1_small_mofid.csv
- N2 adsorption filepath: ./benchmark_datasets/hMOF/mofid/hMOF_N2_1_small_mofid.csv
- Regenerability filepath: ./benchmark_datasets/regenerability/mofid/regenerability_small_mofid.csv

To perform inference and obtain hMOF prediction results, run inference_dataset_generation.py to generate an inference dataset. Then, run inference_compilation.py to organize it. Finally, run transformer_inference.py to generate final predictions. This will create inference_results.csv, a dataset containing adsorption predictions for each hMOF. If you would like to change the property that is predicted, change the data name/path in config_ft_transformer.yaml with the corresponding information:

- CO2 adsorption:
  data_name: 'hMOF_CO2_1_test'
  dataPath: './benchmark_datasets/hMOF/mofid/hMOF_CO2_1_small_mofid.csv'

- N2 adsorption:
  data_name: 'hMOF_N2_1_test'
  dataPath: './benchmark_datasets/hMOF/mofid/hMOF_N2_1_small_mofid.csv'

- Regenerability:
  data_name: 'regenerability_test'
  dataPath: './benchmark_datasets/regenerability/mofid/regenerability_small_mofid.csv'
