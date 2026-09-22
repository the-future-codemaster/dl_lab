# Experiment 5: Study of Weight Initialization, Regularization, Optimization, and Transfer Learning

## Folder Structure
- `Lab5.ipynb`: Source code for Experiment 5 (MobileNetV2 on Oxford-IIIT Pet Dataset).
- `Experiment_5.tex`: LaTeX source code for the experimental report.
- `Experiment_5.pdf`: Compiled PDF report of the experiment.
- `README.md`: This documentation file.

## Objective
To systematically study the effect of weight initialization, regularization, optimization algorithms, CNN hyperparameters, transfer learning, fine-tuning and cross-validation on image classification performance. The experiment uses a single CNN architecture (MobileNetV2) and the Oxford-IIIT Pet dataset to evaluate the effect of individual design choices using training/validation curves and ultimately select a suitable configuration using 5-fold cross-validation.

## Dependencies

### Python Dependencies
- Python 3.x
- TensorFlow / Keras (with `keras.applications` for pretrained MobileNetV2)
- NumPy, Matplotlib, Scikit-learn, OpenCV
- Jupyter Notebook / Google Colab

### LaTeX Dependencies
To compile the `Experiment_5.tex` file, standard LaTeX distribution packages are required:
- `geometry`, `amsmath`, `amssymb`, `graphicx`, `booktabs`, `array`, `float`, `hyperref`, `enumitem`, `xcolor`
- `tikz` (with `positioning, arrows.meta, shapes.geometric, multirow`)

## Execution Instructions
1. Open `Lab5.ipynb` in Jupyter Notebook or Google Colab.
2. Run the notebook to download the Oxford-IIIT Pet dataset and execute the different experimental settings.
3. Compile `Experiment_5.tex` using Overleaf or a local LaTeX distribution. Note: Ensure any generated plots are placed in the same directory before compilation.
