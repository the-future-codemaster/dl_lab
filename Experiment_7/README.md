# Experiment 7: Autoencoders and Variational Autoencoders

## Objective
The objective of this experiment is to develop an end-to-end understanding of autoencoders and their variants for image representation, reconstruction, denoising, and generative modeling using the MNIST dataset.

## Architectures Implemented
1. **Fully Connected Autoencoder (FC-AE)**: Maps the 784-dimensional flattened MNIST image to a 16-dimensional bottleneck and reconstructs it.
2. **Convolutional Autoencoder (CAE)**: Utilizes `Conv2D` and `Conv2DTranspose` / `UpSampling2D` layers to preserve spatial hierarchies, achieving vastly superior reconstruction quality with fewer parameters.
3. **Denoising Autoencoder (DAE)**: Injects Gaussian and Salt-and-Pepper noise into the input images, training the autoencoder to actively remove the noise and restore the original clean digit structure.
4. **Variational Autoencoder (VAE)**: Replaces the deterministic bottleneck with a probabilistic latent space $\mathcal{N}(\mu, \sigma^2)$, regularized by KL Divergence. Enables continuous latent space interpolation and the direct generation of novel synthetic digits.

## Metrics
Models were quantitatively evaluated using:
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Structural Similarity Index Measure (SSIM)

## Contents
* `Lab7.ipynb`: The primary Jupyter Notebook containing the end-to-end Python implementation, training loops, mathematical evaluations, and data visualizations.
* `main.tex`: The LaTeX source code for the final laboratory report.
* `amrut_lab7.pdf`: The finalized, compiled laboratory report detailing the performance, visualizations, and comparative inferences of all implemented autoencoder models.
