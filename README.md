# Uncertainty Quantification for Reinforecment Learning - WIAS Institute
This project is part of Research Project : Uncertainty Quantification for Reinforcement Learning, as a part of PhD research in WIAS


**TODO for CVAE**

1. Experiment with full Cov Gauss posterior vs Factorized ones
ref
Intro to VAE paper sections 2.5 and 2.5.1
https://arxiv.org/pdf/1906.02691.pdf

=======================================================

**Roadmap for CVAE experimentation**

Next steps
This tutorial has demonstrated how to implement a convolutional variational autoencoder using TensorFlow.

As a next step, you could try to improve the model output by increasing the network size. For instance, you could try setting the filter parameters for each of the Conv2D and Conv2DTranspose layers to 512. Note that in order to generate the final 2D latent image plot, you would need to keep latent_dim to 2. Also, the training time would increase as the network size increases.

You could also try implementing a VAE using a different dataset, such as CIFAR-10.

VAEs can be implemented in several different styles and of varying complexity. You can find additional implementations in the following sources:

Variational AutoEncoder (keras.io)
VAE example from "Writing custom layers and models" guide (tensorflow.org)
TFP Probabilistic Layers: Variational Auto Encoder
If you'd like to learn more about the details of VAEs, please refer to An Introduction to Variational Autoencoders.



apply EDA to images https://towardsdatascience.com/exploratory-data-analysis-ideas-for-image-classification-d3fc6bbfb2d2
ordinary analysis
Roadmap

i) latent direciton

1) plot latent distribution

2) experiment with 3 / 4 dimension latents instead of 2

2) experiment more complex latent posterior (Gaussian with covariance structure - review the paper)

3) experiment with exponential distribution (generic)
ii) data direction

1) formalize metrics to discribe distribution

2) compare input output based on compare (distributions) => GenEval paper

 - lower intrinsic dimensionality 

 - multi-modality 

 - compositionality

 -  independence

 - causal structure.

