from Bio.PDB import PDBParser, Superimposer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GT_PATH = BASE_DIR / "data" / "pdb_gt" / "ground_truth.pdb"
PRED_PATH = BASE_DIR / "output" / "prediction.pdb"
OUT_PATH = BASE_DIR / "output" / "evaluation.txt"

parser = PDBParser()
gt_structure = parser.get_structure("GT", GT_PATH)
pred_structure = parser.get_structure("PRED", PRED_PATH)

gt_atoms = [atom for atom in gt_structure.get_atoms() if atom.get_id() == 'CA']
pred_atoms = [atom for atom in pred_structure.get_atoms() if atom.get_id() == 'CA']

min_len = min(len(gt_atoms), len(pred_atoms))
gt_atoms = gt_atoms[:min_len]
pred_atoms = pred_atoms[:min_len]

sup = Superimposer()
sup.set_atoms(gt_atoms, pred_atoms)
sup.apply(pred_structure.get_atoms())

with open(OUT_PATH, "w") as f:
    f.write(f"RMSD: {sup.rms:.4f}\n")
print(f"Done! RMSD: {sup.rms:.4f}")
