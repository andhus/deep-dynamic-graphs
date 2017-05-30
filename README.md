# deep-dynamic-graphs
Some tutorial code put togehter for Talk in Stockholm AI Studdy Group 2017-06-01 on topic:

## Application of RNNs to Multiple Asynchronous Event-Driven Data Streams
Anders Huss, Machine Learning Team Lead at Watty (watty.io)

In most school-book examples, RNNs are applied on a single stream of (typically regular) data such as text or sound/video with a fixed sampling rate. However, reality is full of cases where one need to combine multiple streams of data into a single model - streams that might be event-driven and with great variation in frequency as well as content. This could for example be the case for the control system of a self driving car that receives input from many different sensors, or if we want to apply predictive modelling based on any event-driven micro service system.

In this session we’ll explore how one can apply RNN in such cases. There will be a hands on demo of it can be done end-to-end by utilising the concept of "dynamic computational graphs" with TensorFlow Fold (github.com/tensorflow/fold).

