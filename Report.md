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

![Architecture of text-to-image synthesis](https://imgur.com/K5DzKWu.jpg =450x)
*Architecture of text-to-image synthesis* [^fn1]

### Sentence embedding

We use a pre-trained corpus to encode our sentence. Similar to word2vec, a popular word embedding model, skip-thought vector model uses the sentence itself and its context to train the neural network in order to get the vector. The major difference is that it uses recursive neural networks to deal with sentences. Because training vectors consumes time and requires a large amount of language data, we first try to use the pre-trained model for our sentences.

![Architecture of skip-thought vectors](https://cdn-images-1.medium.com/max/2000/1*MQXaRQ3BsTHpn0cfOXcbag.png "Architecture of skip-thought vectors")
*How sentences encoded into embedded vectors* [^fn2]

### General Architecture

With encoded sentence, we can build our generative neural network using normal deep convolutional layers. Our general architecture is:

![DC-GAN architecture](https://imgur.com/F09PuVw.jpg "DC-GAN architecture")

### Discriminator
For the discriminator, we use three convolutional layers, followed by ReLU activation, batch normalization and max pooling. Finally, it goes through two fully-connected layers and sigmoid functions to the loss function.

### Generator
For the generator, we have tried several different architectures, according to in which stage different components are involved. We tried different stages. For example, we first concatenate all inputs together and let them go though convolutional layers. Our results were not good, and generated images have no diversity although random is involved. So we tried to use L2 normalization and rearrange the input order. We let input image go though some convolutional layers first, then concatenating with sentence vector. We add noise after then to emphasise the weight of noise. We push the combined data though four convolutional layers, followed by two fully-connected layers and tanh activation function. The final structure of our generator is given below, which is also used in some other architectures.

![Architecture of Generator](https://imgur.com/WlsAUgJ.jpg "Architecture of Generator")

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


A Double Discriminator GAN (DD-GAN) approach 
----------------------------------
The issue we faced with the previous network was that there was no information being passed to the discriminator to confirm whether the generated output from the text input was correct solution of the operation or not. Since this text embedding is based on an inference, i.e, the solution of the text is an answer to an arithmetic operation, we realised it essential to have a classifier network to classify the number being generated. 

![](https://imgur.com/1jJbfch.jpg)

The above architecture describes the new network where the generator combines the losses of 2 separate networks:
1) The discriminator which distinguishes between a real and fake sample
2) The classifier network which classifies the generated output/real output images into one of the 100 classes

The generator tries to maximize the discriminator's loss on the generated output and minimize the loss of the classifier on the generated output for the real class. 

The architecture of the generator and discriminator is quite similar to the one explained in DC-GAN. 


### One-hot version of input sentences

Original version of sentence embeddings use pre-trained skip-thoughts vectors, which are slow to run and hard to customize. The pre-trained dataset uses a large corpus of natural language, which makes every vector very long, and not suitable for our case. So we decided to change it to a simpler one. One way is to train the dataset by ourselves using a limited corpus, another way is to just use the one-hot vector to represent our sentence, because currently, our sentences contain limited number of words, and we do not need to determine the semantic relationship between the words. After we successfully train the whole model, we can train our own sentence embedding.

### Training DD-GAN and Results
![](https://imgur.com/aj1u1HI.jpg =240x)![](https://imgur.com/PNaFLWN.jpg =240x)![](https://imgur.com/QGCqXeX.jpg =220x)

It can be seen from the above results that the problem of mode collapse hasn't been solved. 

All numbers being generated look similar so the generator seems to be learning only one kind of solution.

Even though the graphs of the two losses are similar, the network does not seem to learn the images. 


Separate Networks
----------------------------------
As shown from the results (or the lack thereof) above in DDGAN, we thought it would make more sense for us to break the network down into separate parts.

Image generation and operation inference. We have tried to build separate networks for inference and generation and hope to find the issues the networks were struggling with. 

### Operation Inference 

The inference network would take a natural language sentence and try to compute the number implied by the sentence. Say for example, an input sentence 'Multiply 6 by 4' would have to produce an output of 24. 

We tried two kinds of networks for this purpose. The first is a simple feed-forward network. The input to the network is a 'multi-hot' embedding of the sentence and the output of the network is a one-hot representation of the expected solution. 

We chose to create our own embeddings because any pre-trained embedding would be created from a much larger dataset, and therefore much of the information in the embeddings would be unnecessary at best, and a distraction at worst.The 'multi-hot' embedding for the sentence was created as a numpy array of size equal to the size of the vocabulary. In our case, the vocabulary is a small set of unique words in the dataset. The sentence embeddings were created by turning on the bit corresponding to each word in the sentence. 

The feed-forward neural network explained above was trained on 60,000 example for 10 epochs. The training accuracy reached 1 very quickly, suggesting that it overfit the data very quickly. We tested it on the test data, and found the accuracy to be very low. 

So, we trained an RNN with a LSTM cell on the same dataset. That did not improve the accuracy either. 




### Image Generation

In this network, the generator takes in a one hot encoded vector concatenated with noise as an input. The one hot encoded vectors are representations of numbers ranging from 0-99. 

The discriminator takes in the generated output, and real output images to distinguish between what's real and what's fake. 

The generator tries to maximize the discriminator's loss on the generated output and minimizes the discriminator's loss on the real output. The discriminator tries to maximize its on the real output and minimize its loss on the generated output. 

The figure below depicts a simple representation of the GAN used for this purpose. The data split was 60,000 images for training.

![](https://imgur.com/0aAdzG4.jpg =240x) ![](https://imgur.com/aChgpzH.jpg =240x) 

The GAN being used for this purpose is a normal Vanilla GAN  

### Results

Results were to generate a sample of data for the first 64 (Index 0 - 64) data points was set. 


 ![](https://imgur.com/0gL3jed.jpg =240x) ![](https://imgur.com/LgUVF3b.jpg =240x) ![](https://imgur.com/Q3364fO.jpg =220x)

It can be seen that the results are somewhat satisfactory. Even though the exact results haven't been learned, the network has been able to learn rows of 20's, 30's, 40's ,50's, 60's and some of them are in order as well. 

The results are still not entirely satisfactory. We tried several methods like AC-GAN[^fn3], conditional-GAN and DDGAN as seen above. Unfortunately, we were unable to come up with better results with any of them. 


Conclusion and Future Scope
----------------------------------

We have tried many different networks, with different approaches, loss functions, and techniques. However, the results haven't been very satisfactory. The problem is harder than we initially expected. Mode collapse seems to be a challenge that is pending to be solved. This remains to be a work-in-progress. We are determined to continue working on this project even after the course ends. 


[^gan]: Goodfellow, Ian, et al. "Generative Adversarial Networks" arXiv preprint arXiv:1406.2661. 2014.

[^fn0]: LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, 1998.

[^fn1]: Reed, Scott, et al. "Generative Adversarial Text-to-Image Synthesis" Proceedings of The 33rd International Conference on Machine Learning. 2016.

[^fn2]: Kiros, Ryan, et al. "Skip-Thought Vectors" arXiv preprint arXiv:1506.06726. 2015.

[^fn3]: Odena, Augustus, et al. "Conditional Image Synthesis with Auxiliary Classifier GANs" Proceedings of the 34 th International Conference on Machine Learning. 2017.
