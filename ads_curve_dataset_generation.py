import os
import csv
import yaml
import shutil
import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib

import glob
import json
import subprocess
from scipy.optimize import curve_fit

# Define an empty list to store Cif data and properties
data = [[], [], [], [], [], []]

MOFID_SIF = "mofid.sif"

# Define the linear function
def linear(x, K):
  return K * x

# Define the non-linear function
def freundlich(x, K, a, n, b):
    return K * np.power(abs(x - a), 1/n) + b

# Loop through fetched Mofs and extract data
for filename in glob.glob(os.path.join("hMOF_CIFs/", '*.json')):

  mofid_output = ""

  with open(filename, encoding='utf-8', mode='r') as currentFile:
    json_data = json.loads(currentFile.read().replace('\n', ''))
    mofid_output = json_data["mofid"]
  '''
  if mofid_output == None or mofid_output == "* MOFid-v1.NA.NA":
    mofid_cmd = ["singularity", "run", MOFID_SIF, "file", filename.replace(".json", ".cif")]
    mofid_run = subprocess.run(mofid_cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sys.stderr.write(mofid_run.stderr)  # Re-forwarding C++ errors
    mofid_output = json.loads(mofid_run.stdout)["mofid"]
  '''


  if mofid_output != None and "*" not in mofid_output:
    co2_curve = [json_data["isotherms"][0]["isotherm_data"][3]["total_adsorption"], json_data["isotherms"][0]["isotherm_data"][1]["total_adsorption"], json_data["isotherms"][0]["isotherm_data"][4]["total_adsorption"], json_data["isotherms"][0]["isotherm_data"][2]["total_adsorption"]]
    n2_curve = [json_data["isotherms"][1]["isotherm_data"][0]["total_adsorption"], json_data["isotherms"][1]["isotherm_data"][1]["total_adsorption"]]

    # Fit the curve to the sample data
    popt, pcov = curve_fit(linear, [0.09, 0.9], n2_curve, p0=[1])
    # Predict N2 capacity at 1 bar (x = 1)
    n2_capacity = linear(1, *popt)

    try:
      # Fit the curve to the sample data
      popt, pcov = curve_fit(freundlich, [0.05, 0.1, 0.5, 2.5], co2_curve, p0=[1, 0, 1, 0])
      co2_capacity = freundlich(1, *popt)

      if np.isnan(co2_capacity):
        popt, pcov = curve_fit(linear, [0.05, 0.1, 0.5, 2.5], co2_curve, p0=[1])
        co2_capacity = linear(1, *popt)

        # Calculate R-squared for CO2 linear regression
        y_pred_co2 = linear(np.array([0.05, 0.1, 0.5, 2.5]), *popt)
        ss_res_co2 = np.sum((co2_curve - y_pred_co2) ** 2)
        ss_tot_co2 = np.sum((co2_curve - np.mean(co2_curve)) ** 2)
        r_squared_co2 = 1 - (ss_res_co2 / ss_tot_co2)

        # Check if R-squared of CO2 linear regression is below 0.95
        if r_squared_co2 < 0.95:
            co2_capacity = np.nan
            print("ohhhhno")

    except:
      # Fit the curve to the sample data
      popt, pcov = curve_fit(linear, [0.05, 0.1, 0.5, 2.5], co2_curve, p0=[1])
      co2_capacity = linear(1, *popt)

      # Calculate R-squared for CO2 linear regression
      y_pred_co2 = linear(np.array([0.05, 0.1, 0.5, 2.5]), *popt)
      ss_res_co2 = np.sum((co2_curve - y_pred_co2) ** 2)
      ss_tot_co2 = np.sum((co2_curve - np.mean(co2_curve)) ** 2)
      r_squared_co2 = 1 - (ss_res_co2 / ss_tot_co2)

      # Check if R-squared of CO2 linear regression is below 0.95
      if r_squared_co2 < 0.95:
          co2_capacity = np.nan
          print("ohhhhno")

    # Check for experimental errors (decreasing capacity with increasing pressure)
    if any(co2_curve[i+1] < co2_curve[i] for i in range(len(co2_curve) - 1)):
        co2_capacity = np.nan
    data[0].append(filename.replace(".json", "").replace("inference_dataset/", ""))
    data[1].append(mofid_output)

    data[2].append(co2_curve)
    data[3].append(n2_curve)

    data[4].append(co2_capacity)
    data[5].append(n2_capacity)

  print(filename, mofid_output)

adsorption_curve_dataset = pd.DataFrame({"Name": data[0], "MOFid": data[1], "CO2 Adsorption Curve": data[2], "N2 Adsorption Curve": data[3], "CO2 Adsorption STP": data[4], "N2 Adsorption STP": data[5]})
print(len(adsorption_curve_dataset))
joblib.dump(adsorption_curve_dataset, 'hMOF_dataset.pkl', compress='zlib')

print("success.")