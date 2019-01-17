### CSCI-599-Text-to-Image-Synthesis
Image modification with text commands using Generative Adversarial Networks
===================

This blog chronicles our efforts to build a neural network to infer arithmetic calculations from numbers in images.

The idea of our project was inspired by the expanding applications of GAN. One particular paper based on Text to Image Synthesis captured our attention and brewed the motivation to further enhance the paper's application.  Our project focuses on trying to modify or generate meaningful images from text commands using Generative Adversarial Networks(GANs)[^gan]. A GAN is basically two competing networks, continuously trying to better each other. It consists of a generator that tries to generate an output from noise and a discriminator that tries to predict whether the image is real or not. With more training, the generator learns from the discriminator's prediction and tries to produce outputs that are as close to the real ones as possible. The discriminator simultaneously becomes better at distinguishing what's real and what's generated.

The main idea for our project comes from an ICML paper "Generative Adversarial Text to Image Synthesis", which translates visual concepts from characters to pixels. It uses skip-thought vectors- a sentence to vector encoder, to process natural sentences. The sentence embeddings along with noise are sent to the generator to get meaningful images. Real and fake text along with the real image input/generated output are fed into the discriminator. 

We proposed to work on text-based arithmetic operations on images of numbers. Given a text and an input image, the generator network should be able to understand what the text means and generate a pixelated solution of the text and image input. The bigger picture of this idea is to be able to manipulate/edit a picture, given a natural language command and an image input. This application could find use in several image editing softwares.

We break it down to a simple dataset for our purpose to guage whether the idea is feasible or not. We consequently resorted to using the MNIST dataset [^fn0], because the dataset consists of simple, small-sized images with only 1 channel (Black/White). 
Sample input and desired outputs are shown below:

|              Input             | Output                    |
|:------------------------------:|:-------------------------:|
| Text: Multiply the number by 6 | 
| Image: the number ![](https://i.imgur.com/s53Eqgg.jpg "8") |![48](https://imgur.com/VuLLSqV.jpg "48")            |

|              Input             | Output                    |
|:------------------------------:|:-------------------------:|
| Text: Square the number        |
| Image: the number ![](https://i.imgur.com/s53Eqgg.jpg "8") |   ![84](https://imgur.com/iV5rIBQ.jpg "64")            |

