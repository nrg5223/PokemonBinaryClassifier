# Pokemon Legendary Classifier
## Overview
This Jupyter Notebook trains a binary classifier to predict whether a given Pokémon is legendary based on its attributes. The notebook utilizes PyTorch for building and training a neural network model. Key steps include data preprocessing, model training, and model evaluation.

## Dataset
The dataset used is pokemon.csv, containing information on various Pokémon attributes, such as types, abilities, and stats. Additionally, images of Pokémon are used to display and verify model predictions.

## Requirements
- Python 3.x
- Required packages:
  - pandas
  - seaborn
  - scikit-learn
  - torch
  - PIL (Pillow)
  - kagglehub (for downloading images)

## Workflow
1. Data Loading and Exploration:
- Load the dataset and display the structure.
- Plot class distribution of legendary and non-legendary Pokémon.

## Data Preprocessing:
- Select numeric columns only and remove unhelpful features.
- Handle missing values and standardize the data.

## Train-Test Split:
- Split data into training and test sets, ensuring an even distribution of classes.

## Dataset and Dataloader Creation:
- Define PyTorch Dataset classes for training and test data.
- Use DataLoader to efficiently batch and shuffle data.

## Model Definition:
- Build a fully connected neural network with PyTorch.
- The network includes linear, ReLU, batch normalization, and dropout layers.

## Training:
- Define the binary cross-entropy loss function and Adam optimizer.
- Train the model over 20 epochs, logging loss and accuracy for each epoch.

## Evaluation:
- Test the model's accuracy using a confusion matrix.
- Evaluate specific Pokémon predictions, including displaying images for verification.

## Image Display:
- Fetch Pokémon images from the Kaggle dataset and display images of predicted legendary Pokémon.

## Usage
To run the notebook:
1. Clone/download the notebook and dataset files.
2. Install dependencies using pip:

    `pip install pandas seaborn scikit-learn torch pillow kagglehub`

3. Run the notebook, following the cells sequentially.

## Key Files
- `pokemon.csv` : Dataset of Pokémon attributes.
- `pokemon_images/` : Folder containing Pokémon images.

## Results
The model trains over 20 epochs, achieving accuracy on the test set. It uses a confusion matrix for final evaluation. The notebook also allows users to visualize predictions for legendary Pokémon with corresponding images.
