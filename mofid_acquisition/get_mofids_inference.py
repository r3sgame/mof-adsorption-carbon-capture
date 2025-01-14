import joblib
import json
import sys
import subprocess
import glob
import os

# Define an empty list to store Cif data and properties
data = [[], []]

MOFID_SIF = "mofid.sif"

# Loop through fetched Mofs and extract data
for filename in glob.glob(os.path.join("inference_dataset/", '*.json')):
  mofid_output = ""

  with open(filename, encoding='utf-8', mode='r') as currentFile:
    json_data = currentFile.read().replace('\n', '')
    mofid_output = json.loads(json_data)["mofid"]
  
  if mofid_output == None or mofid_output == "* MOFid-v1.NA.NA":
    mofid_cmd = ["singularity", "run", MOFID_SIF, "file", filename.replace(".json", ".cif")]
    mofid_run = subprocess.run(mofid_cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sys.stderr.write(mofid_run.stderr)  # Re-forwarding C++ errors
    mofid_output = json.loads(mofid_run.stdout)['mofid']

  data[0].append(filename.replace(".json", "").replace("inference_dataset/", ""))
  data[1].append(mofid_output)
  print(filename, mofid_output)

joblib.dump(data, 'mofids_inference.pkl', compress='zlib')

print("success.")