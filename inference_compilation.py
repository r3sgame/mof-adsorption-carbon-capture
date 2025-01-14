import joblib
import numpy as np
import pandas as pd

df1 = pd.read_csv('inference_dataset.csv', header=None).iloc[:, 0]
df2 = pd.read_csv('inference_metadata.csv', header=None)
df3 = pd.read_csv('training_results/finetuning/Transformer/Trans_CGCNN_hMOF_CO2_1_1/inference_results.csv', header=None)
df4 = pd.read_csv('training_results/finetuning/Transformer/Trans_CGCNN_hMOF_N2_1_1/inference_results.csv', header=None)
df5 = pd.read_csv('training_results/finetuning/Transformer/Trans_CGCNN_regenerability_1/inference_results.csv', header=None)

full_df = pd.concat([df1, df2, df3, df4, df5], axis=1)
full_df.iloc[:1000000].to_csv('final_predictions_1.csv', index=False, header=False)
full_df.iloc[1000000:].to_csv('final_predictions_2.csv', index=False, header=False)