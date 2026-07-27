# AdaAD

## Adaptive Contrastive Learning with Dual-Level Augmentation and Cross-View Denoising for Unsupervised Anomaly Detection in Time Series
 
[Paper PDF]()
---

## Overview

AdaAD is a novel contrastive learning-based method for unsupervised time series anomaly detection. It introduces three key modules: a dual-level augmentation framework equipped with learnable temporal permutation to enable data-adaptive and task-aware view generation, a cross-view denoising mechanism to suppress view-specific noise from the initial views to enhance anomaly discrepancy, and an adaptive contrastive learning objective to learn normality-invariant representation between views while amplifying the discrepancies of anomalies.

|![Figure1](img/framework-bold.png)|
|:--:| 
| *Figure 1: Overall architecture of the AdaAD.* |
---

## Code Description

```text
AdaAD/
├── data_factory/           # The datasets preprocessing folder and files.
├── dataset/                # The dataset folder.
├── checkpoints/            # Directory for saving trained model checkpoints. The pretrained checkpoints included in this repository can be used directly for testing.
├── metrics/                # The evaluation metrics code folder, which includes VUC, affiliation precision/recall pair, and other common metrics.
│── AdaAD.py                # AdaAD model files. The details are presented in the paper.
│── attention.py            
│── augmentations.py        
├── result/                 # The results and train processing logs are saved in this folder.
├── scripts/                # The experiments scripts. 
├── utils/                  # The functions for data processing.
├── img/                    # Images in readme.md.
├── main.py                 # The main python file. 
├── solver.py               # The training and testing processing file.
└── requirements.txt        # Python dependencies required to run AdaAD.
```

<!-- Add the public download URL after uploading the processed datasets. -->
**Dataset download:** `<DATASET_DOWNLOAD_URL>`

---
## Installation
To install AdaAD from source, you will need the following tools:

- `git`
- `conda` (Optional)

## Python
- Python ==3.12.2, PyTorch ==2.5.1
#### Steps for installation

**Step 1:** Clone the repository using `git` and change into its root directory.

```bash
git clone https://github.com/ShinoMashiru/AdaAD.git
cd AdaAD/
```

**Step 2:** Download the required datasets and weights from the following links,unzip data in the `AdaAD/dataset` folder. 
**Dataset download:** [Google Drive datasets](https://drive.google.com/drive/folders/1_qolFRkGNEr7Nfy3xZKThBxqdv1zvabe?usp=sharing)

```bash
cd AdaAD/dataset
```

**Step 3:** Create and activate a `conda` environment named `adaad`.

```bash
conda env create -f environment.yml
conda activate adaad
```
Alternatively, if you do not use `conda`, create a Python environment and install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Get Start

### Reproduce the Main Results

To directly reproduce the main experimental results, run the corresponding test scripts:

```bash
bash ./scripts/SWaT.sh
bash ./scripts/PSM.sh
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/NIPS_TS_GECCO.sh
bash ./scripts/NIPS_TS_SWAN.sh
```

The trained model checkpoints are provided in the `./checkpoints/` directory, so the scripts can directly load the pretrained models for evaluation.

### Retrain the Models

Each experiment script contains both training and testing commands. For example, the MSL script is organized as follows:

```bash
# Training
# python main.py --anormly_ratio 0.63 --num_epochs 5 \
#   --batch_size 256 --mode train \
#   --dataset MSL --data_path MSL \
#   --channels 55 --win_size 150 \
#   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 \
#   --max_seg 10 --λ 0.0

# Testing
python main.py --anormly_ratio 0.63 --num_epochs 5 \
  --batch_size 256 --mode test \
  --dataset MSL --data_path MSL \
  --channels 55 --win_size 150 \
  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 \
  --max_seg 10 --λ 0.0
```

To retrain a model, uncomment the training command in the corresponding script and run the script again. The newly trained checkpoint will be saved in the `./checkpoints/` directory and can subsequently be used for testing.

### Parameter Sensitivity Experiments

The parameter sensitivity experiments can be reproduced by modifying the corresponding parameter values in the experiment scripts and rerunning them.

For example, parameters such as the window size, augmentation strengths, number of segments, loss coefficient, and anomaly threshold can be adjusted directly in the command-line arguments.

### Ablation Studies

The model components that should be modified or disabled for the ablation studies are indicated by comments in the source code. The corresponding ablation variants can be reproduced by following these comments, modifying the relevant model code, and rerunning the experiment scripts.

---

## Main Results

The following table reports Precision, Recall, and F1-score from the paper.

|![Figure3](img/main_result1.png)|
|:--:| 
| *Overall comparison on five standard benchmark datasets. Best results are in bold and the second ones are underlined.* |

<div align="center">
  <img
    src="./img/main_result2.png"
    width="40%"
    alt="Comparison on NIPS-TS datasets"
  />

  <br>

  <em>
    Comparison on NIPS-TS datasets. Best results are in bold and the second ones are underlined.
  </em>
</div>

|![Figure5](img/main_result3.png)|
|:--:| 
| *Comparison on NIPS-TS datasets under diverse metrics. Best results are in bold and the second ones are underlined.* |


---
