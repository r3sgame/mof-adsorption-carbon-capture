import joblib
import json
import sys
import subprocess

# Define an empty list to store Cif data and properties
try:
  data = joblib.load("mof_vectors.pkl")
except:
  data = []

# CEGDUO

print(data[0])