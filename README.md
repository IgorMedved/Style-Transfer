# Style-Transfer

This project is implementation of fast style transfer network in PyTorch.
The work on this project was inspired by my studies for Computer Vision Nanodegree where the links to the tensorflow repository
https://github.com/lengstrom/fast-style-transfer was provided as one of many examples of cool Artificial Nearal Networks applications.


If you are interested to learn more about the style tranfer theory these articles provide good background:
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), Johnson's [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/), and Ulyanov's [Instance Normalization](https://arxiv.org/abs/1607.08022)

The model was trained on Microsoft Coco dataset.
I used vgg19 network for producing style and content loss functions. And built
3 convolutional - 5 residual blocks - 3 transpose convolutional layer network to be used as a transform network similar to 
described in the reference


Here is an example results of using transform network trained on Monet's Waterlily painting style
![ScreenShot](/data/style/waterlily.jpg)


for stylizing the pool picture in the coco dataset

![ScreenShot](/screenshots/pic.png)
![ScreenShot](/screenshots/result.png)

One interesting result of this project was the realization that the final picture quality can be improved with network
retraining if we gradually increasing ratio of style transfer/content transfer weights.
The explanation for this is that the transform network has a dual job of
both saving the contents of the original picture and also transfering some style information from the style picture. However as
we initialize the tranform network to the random weights the initial output is random noise.
If the ratio of the style transfer weights to content trasfer weights is too high then the network is never encouraged
to learn the correct content transfer from the original picture and the result is random noise stylyzed as a style photo.
However if the ratio of style/content weight is originally set low, the network learns something akin to an identity transformation.
If this network is then retrained with the larger style/content ratio it does not produce noise anymore, but also keeps content
information.
