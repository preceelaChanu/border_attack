# Stealthy Adversarial Border Attack Implementation

This project implements the method proposed in the paper **"Hiding in Plain Sight: Adversarial Attack via Style Transfer on Image Borders"**.

The implementation focuses on generating adversarial examples by perturbing only the image borders, ensuring visual stealthiness via a style transfer fidelity loss, and measuring the Attack Success Rate (ASR) for both targeted and untargeted scenarios.

## Project Structure

* **`main.py`**: Entry point for running attacks and calculating ASR.
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