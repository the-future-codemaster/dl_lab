# Experiment 4: Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

## Folder Structure
- `Lab4.ipynb`: Source code for Experiment 4 (Transfer Learning & Hyperparameter Tuning).
- `Experiment_4.tex`: LaTeX source code for the experimental report.
- `Experiment_4.pdf`: Compiled PDF report of the experiment.
- `README.md`: This documentation file.

## Objective
To study the evolution of deep CNN architectures (LeNet-5, AlexNet, VGG16, ResNet50), understand transfer learning by fine-tuning pre-trained models on the CIFAR-10 dataset, and compare their classification performance.

## Dependencies

### Python Dependencies
- Python 3.x
- TensorFlow / Keras (with `keras.applications` for pretrained models)
- NumPy, Matplotlib, Scikit-learn
- Jupyter Notebook / Google Colab

### LaTeX Dependencies
To compile the `Experiment_4.tex` file, standard LaTeX distribution packages are required:
- `geometry`, `amsmath`, `amssymb`, `graphicx`, `booktabs`, `array`, `float`, `hyperref`, `enumitem`, `xcolor`
- `tikz` (with `positioning, arrows.meta, shapes.geometric`)

## Execution Instructions
1. Open `Lab4.ipynb` in Jupyter Notebook or Google Colab.
2. Mount Google Drive if using Colab to access the CIFAR-10 dataset.
3. Run all cells sequentially to evaluate LeNet, AlexNet, VGG16 (with transfer learning), and ResNet50.
4. The notebook will automatically generate and save the required `.eps` plots for the LaTeX report.
5. Compile `Experiment_4.tex` using Overleaf or a local LaTeX distribution. Note: If Overleaf times out compiling `.eps` files, convert them to `.png` and update the file extensions in the `.tex` document.
