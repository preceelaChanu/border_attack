# Stealthy Adversarial Border Attack Implementation

This project implements the method proposed in the paper **"Hiding in Plain Sight: Adversarial Attack via Style Transfer on Image Borders"**.

The implementation focuses on generating adversarial examples by perturbing only the image borders, ensuring visual stealthiness via a style transfer fidelity loss, and measuring the Attack Success Rate (ASR) for both targeted and untargeted scenarios.


## Preprocessing

The paper requires removing:
 * Grayscale images.
 * Images that the model already misclassifies (to ensure ASR is calculated fairly).
The preprocess.py script performs these checks and saves the valid, normalized image tensors to disk.
Prerequisites:
You need a folder of images and, ideally, an ImageNet validation mapping file (text file with filename class_index). If your images are in class folders (e.g., data/n01440764/img1.jpg), the script can infer labels too.

Example with a flat folder and a mapping file
```
python preprocess.py --data_dir ./data/raw_images --output_dir ./data/processed --val_map ./val_ground_truth.txt --sample_size 5000
```
 * This will output .pt files (preprocessed tensors) and a dataset_meta.json file containing the labels.


## Running the Attack
Now you can run Targeted or Untargeted attacks using the exact same preprocessed data.
Targeted Attack:

Attack specific target class (e.g., class 1)
```
python main.py --data_dir ./data/processed --output_dir ./results/targeted --attack_type targeted --target_class 1
```
Untargeted Attack:
```
python main.py --data_dir ./data/processed --output_dir ./results/untargeted --attack_type untargeted
```

## Calculating Results
The script will report the Attack Success Rate (ASR) based only on the images that were successfully preprocessed (the "3645" equivalent).

## Project Structure

* **`main.py`**: Entry point for running attacks and calculating ASR.
* **`preprocessing.py`**: Script for performing preprocessing exactly like the paper.
* **`attack.py`**: Core implementation of the optimization algorithm and loss functions.
* **`models.py`**: Definitions for the Target Classifier (ResNet50) and Fidelity Model (VGG19).
* **`utils.py`**: Image preprocessing, border manipulation, and auxiliary functions.

## Prerequisites

* Python 3.8+
* PyTorch
* Torchvision
* Numpy
* Pillow
* Scipy (for L-BFGS implementation)

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```
