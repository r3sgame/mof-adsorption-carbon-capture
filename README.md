# Identifying optimal metal-organic frameworks for adsorption-based carbon capture using natural language processing

## [Read the paper](https://docs.google.com/document/d/1zm9vep4UVej5GFiulpjX60IibpaK2EQztXNkPm2Zaeg/edit?usp=sharing)

Excessive carbon dioxide (CO2) emissions from energy usage contribute to climate change, creating severe environmental issues. Carbon capture aims to mitigate these effects by siphoning CO2 from the atmosphere; a promising subset of this technology is adsorption-based carbon capture, a process that uses adsorbents to separate CO2 from other gases with minimal energy usage. Metal-organic frameworks (MOFs) make for effective adsorbents due to their cage-like shape; however, insufficient research has been conducted on finding MOFs with the highest ability to capture/separate CO2 at an acceptable level of reusability, especially at standard temperature and pressure (STP). Fortunately, transformer models have been created that can use natural language processing (NLP) to convert text representations of these materials into features for machine learning. Using this, three models were trained to predict an MOF’s CO2 adsorption, N2 adsorption, and regenerability by fine-tuning MOFormer (Cao, Magar et al., 2023) on three dedicated datasets (all returning an acceptable loss). Then, modified versions of MOFs in the training data were passed back into the model to predict their adsorption metrics. The results were compared with SIFSIX-Cu-i, an existing optimal MOF, and 38 hypothetical MOFs were found to have a higher adsorption capacity and selectivity with an acceptable regenerability value. This study demonstrates a machine learning-powered process that can significantly speed up material discovery for adsorption-based carbon capture; it identifies high-performing MOFs at STP without the need for expensive physical testing.

## Using this architecture

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
