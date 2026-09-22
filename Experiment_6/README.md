# Experiment 6: End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

## Folder Structure
- `Lab6.ipynb`: Source code for Experiment 6 (Sequence Classification, Hyperparameter Tuning & Video Understanding).
- `Experiment_6.tex`: LaTeX source code for the experimental report.
- `Experiment_6.pdf`: Compiled PDF report of the experiment.
- `README.md`: This documentation file.

## Objective
To develop an end-to-end understanding of recurrent sequence learning by implementing and comparing Vanilla RNN, LSTM and GRU models on the UCI HAR dataset, and extend the pipeline to video understanding using CNN feature extraction (MobileNetV2) on UCF101 followed by an LSTM or GRU.

## Dependencies

### Python Dependencies
- Python 3.x
- TensorFlow / Keras (with `keras.applications` for pretrained models)
- NumPy, Matplotlib, Scikit-learn, OpenCV
- Jupyter Notebook / Google Colab

### LaTeX Dependencies
To compile the `Experiment_6.tex` file, standard LaTeX distribution packages are required:
- `geometry`, `amsmath`, `amssymb`, `graphicx`, `booktabs`, `array`, `float`, `hyperref`, `enumitem`, `xcolor`

## Execution Instructions
1. Open `Lab6.ipynb` in Jupyter Notebook or Google Colab.
2. Ensure the `UCI HAR Dataset` and `UCF101_subset` folders are in the same directory.
3. Run all cells sequentially to evaluate RNN, LSTM, and GRU models.
4. The notebook will automatically generate and save the required `.eps` plots for the LaTeX report.
5. Compile `Experiment_6.tex` using Overleaf or a local LaTeX distribution. Note: If Overleaf times out compiling `.eps` files, convert them to `.png` and update the file extensions in the `.tex` document.
