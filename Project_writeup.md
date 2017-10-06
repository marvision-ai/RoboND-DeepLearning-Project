# Deep learning - Follow Me Project

## Introduction
This project is about training a fully-convolutional neural network to enable a drone to track and follow behind the "hero" target. In this report I will quickly go over network architecture used, training process and final results. It's been quite the struggle!

## Network Architecture 
Before this introduction to fully-convoluted networks, neural nets that include at least some portion of fully connected layers proved to be a good way to achieve reasonable classfication models of entire pictures. 

When using a FCN, you can identify "where" in the image a certain object is. This is done by keeping spacial information through each pass of the filters. I followed the suggestions in the lectures and the project descriptions, and built a Fully Convolutional Network (FCN) to achieve the segmentation needed for the project.

The main building blocks of an FCN are the following:
- Convolutional Neural Network (CNN)
- 1x1 Convolution
- Transposed convolutional layers for upsampling
- Skip connections

These building blocks make up a very good model for image segmentation. The first three building blocks can be seen in this image:
