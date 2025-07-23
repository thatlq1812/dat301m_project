
# Unified Applications of Generative and Transformer-Based Models Across Modalities

This repository contains the final project for the course **DAT301m - Deep Learning Project** (Summer 2025) at FPT University. The project evaluates how generative and transformer-based deep learning models perform across three different data modalities: audio, video, and biological sequences.

## Project Objective

We aim to investigate the transferability, robustness, and limitations of high-capacity AI models when applied outside their original domain. The project includes three representative tasks:

- **Music Source Separation** using **Demucs**
- **Real-Time Video Matting** using **MODNet**
- **Protein Structure Prediction** using **AlphaFold2**

All experiments were conducted on a standard GPU laptop setup with limited compute resources, to simulate real-world deployment scenarios in academia or startups.

## Repository Structure

```
dat301m_project/
├── alphafold/ # AlphaFold2 wrapper code
├── alphafold_task/ # Main script for AlphaFold2 task
├── checkpoints/ # Model checkpoints (e.g., demucs)
├── data/ # Input datasets and FASTA/MSA files
├── demucs/ # Demucs source separation scripts
├── modnet/ # MODNet matting scripts
├── outputs/ # Model outputs (audio, images, etc.)
├── pdb_out/ # AlphaFold2 PDB output files
├── report/ # Final report in LaTeX and PDF
├── runs/ # Logs and runtime data
├── test/ # Evaluation or test samples
├── training_log.csv # Training history (e.g., for Demucs)
├── README.md # Project overview (this file)
```

## How to Run

**Step 1: Clone the repository**

```bash
git clone https://github.com/thatlq1812/dat301m_project.git
cd dat301m_project
```

**Step 2: Install dependencies**  
(Use separate virtual environments if needed)

### Demucs (PyTorch)

```bash
cd demucs
pip install -r demucs_requirements.txt
python evaluate.py
```

Outputs will be saved to `demucs/output/`.

### MODNet (TensorFlow)

```bash
cd modnet
pip install -r tf-gpu_requirements.txt
python evaluate.py
```

Outputs will be saved to `modnet/output/`.

### AlphaFold2 (Protein Structure)

AlphaFold2 requires a separate installation via Docker or Conda.

Refer to:  
https://github.com/deepmind/alphafold

This repository provides a simplified wrapper script. Predicted `.pdb` files will be saved in `data/pdb_out/`.

## Sample Results

| Task                  | Metric        | Result           |
|-----------------------|---------------|------------------|
| Demucs (Vocals SDR)   | SDR            | -26.15 dB         |
| MODNet (Video Matting)| IoU            | 86.1%             |
| AlphaFold2 (Protein)  | RMSD           | 3.1 Å             |

> Note: Results were obtained under constraints of limited training and hardware.

## Report

The final report follows the NeurIPS 2024 format and includes:

- Introduction and Motivation  
- Detailed methodology for each model  
- Evaluation metrics and results  
- Discussion on failure cases and future directions

Access the full PDF here:  
`./report/final_report_neurips.pdf`

## Team Members

- **Hoang Phuc Trong** — SE183203  
- **Le Quang That** — SE183256  
- Class: AI1801  
- Course: DAT301m (Deep Learning Project)  
- Semester: Summer 2025  
- University: FPT University  

## License

This project is provided for academic purposes only.  
We build upon open-source implementations of Demucs, MODNet, and AlphaFold2.  
Refer to their respective repositories for licensing and citation.

## Link

GitHub: https://github.com/thatlq1812/dat301m_project
