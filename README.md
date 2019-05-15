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


Dataset
-------------
The dataset we have employed for this project is the MNIST dataset. We have used opencv to read, modify and concatenate the images, creating datasets of different sizes, and forms for the different networks we ran. 

We automatically generate a double digit dataset for numbers ranging from 0-99 by making use of the original MNIST dataset. Each of the numbers have a fixed number of hand written digits associated with it, which can be set by the user. The double digits are formed by randomly concatenating and resizing two handwritten digits to the same size as the MNIST images. i.e 28x28x1. 

We have considered different types of sentences, for different arithmetic operations, like multiplication, addition and squaring, to diversify the kinds of sentences used. We have created the dataset using these diverse set of sentences, for different kinds of MNIST images. 

The data split we used for the generation was 60,000 sentences for training, and 10,000 sentences for testing.


Text to image synthesis using DC-GAN
----------------------------------
At the beginning, we referred to the architecture from the "Generative Adversarial Text to Image Synthesis" paper. They use skip-thought vector to encode natural language, along with noise to go though generator, built by a deep convolutional neural network. The architecture is given below.

![Architecture of text-to-image synthesis](https://imgur.com/K5DzKWu.jpg =450x) </br>
*Architecture of text-to-image synthesis* [^fn1]

### Sentence embedding

We use a pre-trained corpus to encode our sentence. Similar to word2vec, a popular word embedding model, skip-thought vector model uses the sentence itself and its context to train the neural network in order to get the vector. The major difference is that it uses recursive neural networks to deal with sentences. Because training vectors consumes time and requires a large amount of language data, we first try to use the pre-trained model for our sentences.

![Architecture of skip-thought vectors](https://cdn-images-1.medium.com/max/2000/1*MQXaRQ3BsTHpn0cfOXcbag.png "Architecture of skip-thought vectors")
*How sentences encoded into embedded vectors* [^fn2]

### General Architecture

With encoded sentence, we can build our generative neural network using normal deep convolutional layers. Our general architecture is:

![DC-GAN architecture](https://imgur.com/F09PuVw.jpg "DC-GAN architecture") </br>

### Discriminator
For the discriminator, we use three convolutional layers, followed by ReLU activation, batch normalization and max pooling. Finally, it goes through two fully-connected layers and sigmoid functions to the loss function.

### Generator
For the generator, we have tried several different architectures, according to in which stage different components are involved. We tried different stages. For example, we first concatenate all inputs together and let them go though convolutional layers. Our results were not good, and generated images have no diversity although random is involved. So we tried to use L2 normalization and rearrange the input order. We let input image go though some convolutional layers first, then concatenating with sentence vector. We add noise after then to emphasise the weight of noise. We push the combined data though four convolutional layers, followed by two fully-connected layers and tanh activation function. The final structure of our generator is given below, which is also used in some other architectures.

![Architecture of Generator](https://imgur.com/WlsAUgJ.jpg "Architecture of Generator") </br>

### Training Conditioned DC-GAN
DC-GAN's training is quite fast with the help of GPU. However, as you can see below, the training was not successful. Generated graphs are hard to recognize as digits, and what's worse is there is a lack of diversity even upon adding noise as one of inputs.

Here is diagram of our training loss, along with one output after the final epoch.

![](https://imgur.com/BTwT9f1.jpg =240x) ![](https://imgur.com/KLhWzbx.jpg =240x) ![](https://imgur.com/ieDdvzU.jpg =220x)

### Discussion and next model
In order to introduce some diversity in the output, several tricks were tried such as
1) Adding L2 normalization
2) Using Wasserstein distance in loss function
3) Iterative batch size reduction
4) Make the generator stronger as discriminator loss was going to zero. 

However, the problem of mode collapse still persisted where the network was only learning one type of output. Besides, we also noticed that there was a problem with the model. The discriminator tries to guess whether the image is real or generated. But, it has no way to guess whether it is the number that is expected. It could be any one of the hundred numbers and the discriminator would accept it as a number. So, there was a need for a differnt model. 
