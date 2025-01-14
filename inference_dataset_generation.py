import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

inorganic_layers = ["Cu", "Mn", "Zn", "Pd", "Cr", "Pt", "Fe", "Ni", "Be", "Mg", "Ca", "Sr", "Ba"]
halogens = ["Cl", "Br", "F", "I"]
mod_status = []
mofids = []
og_data = joblib.load('hMOF_dataset.pkl')

for i in range(len(og_data['MOFid'])): #only process .JSON files in folder.
    mofid = og_data['MOFid'][i]

    present_inorganic_layer = ""
    present_halogen = ""

    for inorganic_layer in inorganic_layers:
        if inorganic_layer in mofid.split()[0]:
            present_inorganic_layer = inorganic_layer

    for halogen in halogens:
        if halogen in mofid.split()[0]:
            present_halogen = halogen

    if present_halogen == "" or present_inorganic_layer == "" or "*" in mofid:
        print("mofid cannot be processed")
    else:
        for inorganic_layer in inorganic_layers:
            for halogen in halogens:
                alt_mofid = mofid.split()[0].replace(present_halogen, halogen).replace(present_inorganic_layer, inorganic_layer) + " " + mofid.split()[1]

                if alt_mofid == mofid:
                    mod_status.append("Original")
                else:
                    mod_status.append("Modified")
                mofids.append(alt_mofid.replace(' ', '&&').replace('MOFid-v1.', ''))

                print(i, alt_mofid)

data = pd.DataFrame({"Mofid": mofids, "Value": np.zeros(len(mofids))})
metadata = pd.DataFrame({"Status": mod_status})
data.to_csv('inference_dataset.csv', index=False, header=False)
metadata.to_csv('inference_metadata.csv', index=False, header=False)