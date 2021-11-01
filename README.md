# Example of Semi Supervised GAN

In this episode of my AI tutorials, I implemented a Semi Supervised GAN from scratch to train a classifier in a dataset that contains a small number of labled samples and a much larger of unlabled samples.

Basic knowledge of Keras, and the GAN framework is required.

The Semi supurvised GAN is an extension of the GAN architenture. It is a combination of a GAN and a classifier. The classifier is trained to classify the unlabled samples. The GAN is trained to generate samples that are classified as the same as the labled samples.