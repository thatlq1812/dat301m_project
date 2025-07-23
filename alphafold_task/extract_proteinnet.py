import tarfile
from pathlib import Path

from matplotlib.pylab import sample
import lmdb
import pickle
from pathlib import Path
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

def extract_proteinnet_tar():
    tar_path = Path("data/pdb_gt/proteinnet.tar.gz")
    extract_dir = tar_path.parent / "proteinnet"  # data/pdb_gt/proteinnet

    # Create the extraction directory if it doesn't exist
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tar_path} to {extract_dir}...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    print("Extraction complete.")

def extract_fasta_from_lmdb(lmdb_path: Path, out_fasta_path: Path, max_samples=10):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        records = []

        for i, (key, value) in enumerate(cursor):
            if i >= max_samples:
                break
            sample = pickle.loads(value)

            if "primary" not in sample or "id" not in sample:
                continue

            seq = str(sample["primary"])
            raw_id = sample["id"]
            seq_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)

            record = SeqRecord(Seq(seq), id=seq_id, description="")
            records.append(record)

        SeqIO.write(records, out_fasta_path, "fasta")
        print(f"Saved {len(records)} sequences to {out_fasta_path}")

# This script extracts sequences from the ProteinNet LMDB database and saves them to a FASTA file.

if __name__ == "__main__":
    EXTRACT = False  # Set to True to extract the tar file
    if EXTRACT:
        print("Extracting ProteinNet data...")
        extract_proteinnet_tar()
    lmdb_dir = Path("data/pdb_gt/proteinnet/proteinnet/proteinnet_train.lmdb")
    out_fasta = Path("data/input.fasta")
    extract_fasta_from_lmdb(lmdb_dir, out_fasta)
