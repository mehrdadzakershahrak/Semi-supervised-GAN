# Semi-Supervised GAN

This repository contains the implementation of a Semi-Supervised Generative Adversarial Network (GAN). The GAN is trained using a semi-supervised learning technique, where the discriminator is taught to make predictions on both supervised (labeled) and unsupervised (unlabeled) data.

## Description

The code provided defines and compiles various models of GAN, including standalone discriminator models, supervised and unsupervised discriminator models, and a semi-supervised discriminator model. It also contains the code for defining a standalone generator model, as well as for the combined generator and discriminator model, used for updating the generator. The code includes functions for loading the training images, selecting supervised samples from the dataset, generating real and fake samples, summarizing the performance of the models, and for training the generator and discriminator.

## Code Structure

The code is structured as follows:

1. **Standalone Discriminator Model**: The first part of the code defines a standalone discriminator model using Keras, with architecture details visualized using `plot_model`.

2. **Supervised and Unsupervised Discriminator Models**: The code further defines supervised and unsupervised discriminator models. The supervised model is compiled with 'sparse_categorical_crossentropy' loss function and accuracy metrics. Both models' structures are visualized and saved as '.png' files.

3. **Multioutput Discriminator Model**: A discriminator model with multiple outputs is defined next. It is compiled with both 'binary_crossentropy' and 'sparse_categorical_crossentropy' loss functions, and accuracy metrics.

4. **Semi-Supervised Discriminator Model**: The semi-supervised discriminator model is defined using a custom activation function. It includes both supervised and unsupervised outputs.

5. **Standalone Generator Model**: The generator model is defined with 'tanh' activation function for the output layer.

6. **Combined Generator and Discriminator Model**: A combined model is defined for updating the generator. The weights in the discriminator are made non-trainable.

7. **Training Data Loading and Preparation**: Functions for loading real samples, selecting supervised samples from the dataset, generating real and fake samples, and generating points in the latent space as input for the generator are provided.

8. **Model Performance Summarization**: A function to summarize the performance of the models and save the models and a plot of generated images is provided.

9. **Training Function**: Finally, a function to train the generator and discriminator is provided. The function includes the code to update the supervised discriminator, unsupervised discriminator, and generator.

## Usage

To use this code, simply clone the repository and execute the Python scripts. You can modify the parameters according to your requirements. Ensure that you have all the necessary dependencies installed.

## Dependencies

- Keras
- TensorFlow
- NumPy
- Matplotlib

Note: The code uses the MNIST dataset, which can be directly loaded using the Keras datasets API. If you wish to use a different dataset, you will need to modify the `load_real_samples()` function.

## Contribution

Contributions are always welcome. Please fork this repository and open a pull request to propose your changes.
