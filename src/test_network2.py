##import mnist_loader 
##training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
##import network2 
##net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost) 
##net.large_weight_initializer()
###net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
###net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, lmbda = 0.1, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
##net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)


import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([FullyConnectedLayer(n_in=784, n_out=30), SoftmaxLayer(n_in=30, n_out=10)], mini_batch_size)
net.SGD(training_data[:1000], 60, mini_batch_size, 0.1, validation_data[:1000], test_data[:1000])

if False:
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20*12*12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)   



