# Image-Classification-DNN

This project implements a Deep Neural Network (DNN) from scratch to classify images into specified categories. The model is trained to recognize specific objects in images, such as distinguishing between images of cats and non-cats. This implementation uses a neural network with dimensions [12288, 20, 7, 5, 1] and is trained for 2400 iterations.

## Project Structure

- `DNN_Model.py`: Main script to train and test the DNN model.
- `DNN_Func.py`: Contains the functions used for building, training, and testing the model.
- `datasets/`: Directory containing the training and testing images in h5 format.


### Neural Network Architecture

The DNN model uses the following layer dimensions:
- Input layer: 12288 (corresponding to the flattened 64x64x3 image)
- Hidden layers: [20, 7, 5]
- Output layer: 1 (binary classification)

### Training

- The model is trained for 2400 iterations.
- The training process involves forward propagation, cost computation, backward propagation, and parameter updates.

### Results

- After training, the model evaluates its performance on the test set and provides accuracy metrics.

## Example

Ensure your dataset is correctly placed in the `datasets` folder, and then simply run:

```bash
python DNN_Model.py
