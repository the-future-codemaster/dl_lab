# Experiment 7: End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and Variational Autoencoders

## Objective
The primary objective of this experiment is to construct, train, and evaluate a variety of autoencoder models to understand their capabilities in image reconstruction, denoising, and generative modeling. We study Fully Connected Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and Variational Autoencoders (VAEs) on the MNIST dataset, ultimately performing latent space visualization and interpolation.

## Folder Structure
- `Lab7.ipynb`: A unified Jupyter Notebook containing the entire Python execution pipeline for building, training, and visualizing the autoencoders.
- `Experiment_7.tex`: The finalized LaTeX source file containing the laboratory report, with embedded performance metrics and detailed inferences.
- `Experiment_7.pdf`: The compiled PDF format of the laboratory report.

## Dependencies
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-image (for optional baseline metrics comparisons, though tf.image is also sufficient)

## Execution Instructions
1. Open `Lab7.ipynb` in any Jupyter Notebook environment (e.g., JupyterLab, VS Code, Google Colab).
2. Ensure that the required dependencies are installed (`pip install tensorflow numpy matplotlib`).
3. Run all cells sequentially. The dataset (MNIST) will be fetched automatically via Keras if not already cached.
4. The notebook will automatically generate and display the required 600 DPI `eps` plots and performance metrics tables.
