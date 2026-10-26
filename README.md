# Acute Leukemia Detection

A deep learning project for automated detection of acute lymphoblastic leukemia (ALL) from microscopic blood smear images. This repository contains comparative experiments with pretrained CNNs, an advanced stain-normalized pipeline, and model explainability using Grad-CAM and Grad-CAM++.

## Overview

The goal of this project is to classify peripheral blood smear images into:

- Blast (ALL)
- Non-Blast (Healthy)

The implementation combines:

- Transfer learning with pretrained PyTorch models
- Data preprocessing and augmentation
- Train/validation/test splitting
- Model evaluation using accuracy, precision, recall, F1-score, specificity, and ROC-AUC
- Explainability via Grad-CAM visualizations

## Why this project matters

Acute leukemia is a serious medical condition that requires timely and accurate diagnosis. Computer-aided analysis can support clinicians by highlighting suspicious cell regions and improving consistency in screening workflows.

This repository focuses on image-based classification using standard benchmark datasets such as ALL-IDB1 and ALL-IDB2.

## Repository structure

```text
Acute-Leukaemia-Detection/
├── Leukemia_Detection_PyTorch_Comparative.ipynb   # Comparative deep learning experiments
├── Leukemia_Resnet50.ipynb                        # ResNet50-focused notebook
├── Leukemia_efficient_net.ipynb                   # EfficientNet-focused notebook
├── utilities.py                                   # Shared utilities for dataset loading and modeling
├── proposed/
│   ├── Leukemia_Advanced_Pipeline.ipynb           # Advanced stain-normalized model pipeline
│   └── utilities.py                               # Advanced utilities for CBAM + ArcFace approach
├── base paper/
│   └── An_Efficient_Detection_and_Classification_of_Acute_Leukemia_Using_Transfer_Learning_and_Orthogonal_Softmax_Layer-Based_Model.pdf
├── README.md
└── ...
```

## Included models and experiments

This repository includes multiple modeling strategies:

- ResNet50
- VGG16-BN
- DenseNet121
- EfficientNet-B0
- Advanced pipeline with:
  - stain normalization (Macenko/Reinhard)
  - CBAM attention
  - ArcFace classifier
  - Grad-CAM++ explainability

## Key features

- Image dataset loading and label parsing for ALL-IDB1 and ALL-IDB2
- Stratified train/validation/test splits
- Data augmentation for robust training
- Pretrained backbone adaptation for binary classification
- Metrics reporting and confusion matrix analysis
- ROC-AUC visualization
- Grad-CAM and Grad-CAM++ overlays to interpret model attention
- Checkpointing and resume support for long training runs

## Dataset

The project expects microscopy datasets organized like this:

```text
ALL_IDB1/
└── im/
    ├── image_001_1.jpg
    ├── image_002_0.jpg
    └── ...

ALL_IDB2/
└── img_png/
    ├── image_001_1.png
    ├── image_002_0.png
    └── ...
```

Labels are parsed from filenames using the convention:

- 1 = Blast (ALL)
- 0 = Non-Blast (Healthy)

## Environment setup

This project is built with Python and PyTorch.

### Requirements

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn pillow scikit-learn notebook jupyter
```

### Recommended setup

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

If a requirements file is not present, install the packages listed above manually.

## Running the project

Open the notebooks in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Then open one of the following:

- `Leukemia_Detection_PyTorch_Comparative.ipynb`
- `Leukemia_Resnet50.ipynb`
- `Leukemia_efficient_net.ipynb`
- `proposed/Leukemia_Advanced_Pipeline.ipynb`

## Typical workflow

1. Load and combine ALL-IDB1 and ALL-IDB2 images
2. Apply resizing and augmentation transforms
3. Split data into training, validation, and test sets
4. Train pretrained CNN backbones
5. Evaluate using classification metrics
6. Visualize explanations with Grad-CAM or Grad-CAM++
7. Save checkpoints and compare model performance

## Notes on methodology

The project explores both conventional transfer-learning baselines and more advanced medical-image strategies:

- robust preprocessing for microscopy image variability
- stain normalization to reduce dataset-specific color drift
- attention mechanisms to focus on diagnostically relevant regions
- classification heads tuned for binary leukemia detection

## Project status

This repository is intended for research, experimentation, and educational use in acute leukemia detection using deep learning.

## Reference material

A relevant paper is included under the `base paper` directory:

- `An_Efficient_Detection_and_Classification_of_Acute_Leukemia_Using_Transfer_Learning_and_Orthogonal_Softmax_Layer-Based_Model.pdf`

## Contributing

Contributions are welcome. If you want to improve performance, add reproducibility scripts, integrate a better dataset pipeline, or improve documentation, feel free to open a pull request.

## License

This repository does not currently specify a license file. Please check with the repository owner before reusing the code or data in commercial or publication contexts.

## Acknowledgements

This project builds on standard PyTorch computer vision workflows and publicly available leukemia image datasets. It is designed for academic experimentation and practical model development.

---

If you'd like, I can also tailor the README to a more academic style, a polished GitHub-project style, or a research-poster style version.
