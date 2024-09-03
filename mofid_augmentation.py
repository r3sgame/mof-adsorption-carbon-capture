from openbabel import pybel
import random

def generate__multiple_smiles(smiles, num_variants=5):
  """Generates multiple SMILES strings for a given molecule.

  Args:
    smiles: Th e input SMILES string.
    num_variants: The number of SMILES variants to generate.

  Returns:
    A list of SMILES strings.
  """

  mol = pybel.readstring("smi", smiles)
  variants = []

  for _ in range(num_variants):
    # Randomly permute atom order (might not always produce different SMILES)
    random.shuffle(mol.OBMol.Oatoms)

    # Convert back to SMILES
    new_smiles = mol.write("smi")
    variants.append(new_smiles.strip())

  return variants

# Example usage:
smiles = "c1ccccc1"
variants = generate_multiple_smiles(smiles, 10)
print(variants)