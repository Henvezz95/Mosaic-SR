# Mosaic-SR: Adaptive Super-Resolution for 2D Barcodes
> **Official implementation of the ICIP 2025 paper "MOSAIC-SR: An Adaptive Multi-Step Super-Resolution Method for Low-Resolution 2D Barcodes"**

This repository contains the code and trained models for Mosaic-SR, a multi-step, adaptive super-resolution (SR) method designed to enhance 2D barcode images (e.g., QR codes, Datamatrix). Mosaic-SR devotes more computational effort to regions with barcodes while minimizing effort on uniform backgrounds. The method predicts an uncertainty value for each patch to determine the number of refinement steps required for optimal quality.

## 📝 Paper and Presentation

* **Paper (Open Access):** [https://federicobolelli.it/pub_files/2025icip.pdf](https://federicobolelli.it/pub_files/2025icip.pdf)
* **Presentation Slides (ERASMUS+ Summer School 2025):** [Download Here](https://site.unibo.it/mml-imaging/en/seminars/seminar_vezzali_04_june_2025.ppsx/@@download/file/Seminar_Vezzali_04_June_2025.ppsx)
* **Dataset:** [Download Here](https://ditto.ing.unimore.it/barber/)

![Mosaic-SR Logo](./Mosaic-logo.png)

## 📜 Citing our Work

If you find this work useful in your research, please consider citing our paper:

### BibTeX
```bibtex
@inproceedings{vezzali2025mosaic,
  title={Mosaic-SR: An Adaptive Multi-step Super-Resolution Method for Low-Resolution 2D Barcodes},
  author={Vezzali, Enrico and Vorabbi, Lorenzo and Grana, Costantino and Bolelli, Federico and Datalogic, SpA},
  booktitle={Proceedings of the 2025 IEEE International Conference on Image Processing},
  year={2025}
}
```
## 🔧 Changelog

**November 2026 — fix in the iterative inference.** With the fix, the released models reproduce the paper's decoding results (M1 only: 3,334 vs 3,330; thresholds (0.13, 0.13): 3,529 vs 3,517; (0.05, 0.07): 3,744 vs 3,714, over 9,562 LR/HR pairs; the small differences are due to the zbar version). `test/test_iterative_processing.py` checks the optimized inference against the aggregate Keras model:

```bash
python test/test_iterative_processing.py <path/to/aggregate_model.h5>
```

Dataset building (`buildTFDataset/`) now also accepts both the per-source and the unified BarBeR annotation formats (`QR`/`QR Code`, `DATAMATRIX`/`Data Matrix`, `AZTEC`/`Aztec`), skips regions whose PPE is -1 (number or string), replaces polygons with more than 4 points (curved codes) by their minimum-area rectangle, and gives no crops for images without an annotation entry instead of failing. PyTorch and ONNX are now imported only by the functions that use them.

# 🚀 Getting Started
### Repository Structure
```graphql
mosaic-sr/
├── 3rd-party/  
│   └──  ...                                # Contains reference models for comparison
├── buildTFDataset/
│   ├── create_dataset.py                   # Builds a TFRecord dataset from annotations and images 
│   └── functions.py                        # Additional functions
├── config/
│   └──  ...                                # Contains configuration files in YAML format
├── train/
│   ├── train_pytorch_models.py             # Trains multiple Pytorch models (reference models)
│   ├── train_iterative_models.py           # Trains one or more Mosaic-SR models (ours)
│   ├── training_config_function.py         # Contains training hyperparameters, like batch size, lr-scheduling, etc.
│   └──  ...
├── test/
│   ├── Test Quality - Multi.py             # Measures PSNR, SSIM, and Decoding Rate of reference models
│   ├── Test Quality - Multi Iterative.py   # Measures PSNR, SSIM, and Decoding Rate of Mosaic-SR models
│   ├── Test Time - Multi.py                # Measures Processing time of reference models
│   └── Test Time - Multi Iterative.py      # Measures Processing time of Mosaic-SR models
└── utils/
    └──  ...                                # Contains utility functions
```

### Build Repository
Mosaic-SR upscales images patch by patch. An optimized [im2col](https://github.com/Henvezz95/im2col_2D) function must be built to convert the image to patches. To do that you must call the CMakeLists file in the im2col folder.  

```bash
# Clone the im2col dependency if not already present
# cd mosaic-sr
# git submodule update --init --recursive

cd im2col_2D
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cd ../..
```

CMake will detect your CPU architecture and automatically compile the corresponding SIMD implementation.
Finally, install the required Python libraries.
```bash
pip install -r requirements.txt
```

# 🛠️ Usage

### 0. Download Trained Models
To reproduce the exact results shown in the paper, download the trained models from [here](https://unimore365-my.sharepoint.com/:f:/g/personal/319554_unimore_it/EvX00yibh_1FhSxN_m8cYHsBVYnna4--NamdGXx9eIysNg?e=kE5gnY).

### 1. Prepare the Dataset
Download the BarBeR dataset from [here](https://ditto.ing.unimore.it/barber/) and select "Download Dataset". Extract the dataset from the zip, you will find two folders inside: "Annotations" and "dataset". 
Now, we want to generate a TF Record for each dataset split (training, validation, and test sets). To do that, run the script `create_dataset.py` inside buildTFDataset. The script takes input from a configuration file and an index. The configuration file must be in YAML format, and an example is `config/dataset_config.yaml`. The index is used for K-fold cross-validation. If the configuration file does not select k-fold cross-validation, you can use any index. To build the TFRecord, run this command:
```bash
python ./buildTFDataset/create_dataset.py -c config/dataset_config.yaml -k 0
```
### 2. Train Models
To train the reference models, use `train/train_pytorch_models.py`. To train our proposed pipeline, use `train/train_iterative_models.py`. It will generate a single aggregate CNN model that will be reparameterized into fully connected models M1, M2, and M3 during inference.

### 3. Test Models
The test scripts are inside the folder `/test`. `Test Quality - Multi.py`, is used to measure PSNR, SSIM, and Decoding Rate of reference models, while `Test Quality - Multi Iterative.py` does the same for our proposed pipeline. In the same way, `Test Time - Multi.py` and `Test Time - Multi Iterative.py` can be used to benchmark processing times for reference models and our proposed pipeline, respectively.
