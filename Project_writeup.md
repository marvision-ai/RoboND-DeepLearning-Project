# Deep learning - Follow Me Project

## Introduction
This project is about training a fully-convolutional neural network to enable a drone to track and follow behind the "hero" target. In this report I will quickly go over network architecture used, training process and final results. It's been quite the struggle!

## Network Architecture 
Before this introduction to fully-convoluted networks, neural nets that include at least some portion of fully connected layers proved to be a good way to achieve reasonable classfication models of entire pictures. 

When using a FCN, you can identify "where" in the image a certain object is. This is done by keeping spacial information through each pass of the filters. I followed the suggestions in the lectures and the project descriptions, and built a Fully Convolutional Network (FCN) to achieve the segmentation needed for the project.

The main building blocks of an FCN are the following:
- Convolutional Neural Network (CNN) with Encoder and Decoder constructed with seperable convolution layers
- 1x1 Convolution to connect the encoder and decoder
- Transposed convolutional layers for upsampling
- Skip connections

#### Encoder ####

The encoding stage exsists to extract features that will be useful for segmentation from the specific image. It does this via multiple layers that start by finding simple patterns in the first layer, and then gradually learns to understand more and more complex shapes/structures/feautures in each image the deeper the network goes. This is why I chose to do a 5 layer network; to allow it to increase the features it can find when presented with the training data. 

The Encoder in the FCN will essentially require separable convolution layers, due to its advantages as explained above.  
Below is the implementation for seperable_2d_batchnorm which includes ReLU activation applied to the layers to non-linearize the functions.

```python 
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

```

```python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```


### 1x1 Convolution ###

The 1x1 convolution layer in the FCN, however, is a regular convolution. Below is the implementation of the regular convolution with ReLU activation.

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```


#### Decoder ####
```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([upsampled, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    output_layer1 = separable_conv2d_batchnorm(concat_layer, filters)
    output_layer2 = separable_conv2d_batchnorm(output_layer1, filters)
    
    return output_layer2
```



### Transposed Convolitional Layers for Upsampling ###

```python

```

### Skip connections ###
```python

```






These building blocks make up a very good model for image segmentation. The first three building blocks can be seen in this image:
