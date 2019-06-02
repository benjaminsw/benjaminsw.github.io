---
permalink: /nn/Initialisation/
header:
  image: "/images/digital-transition2.jpg"
---
<h2 id="top"></h2>

### Initialisation: "Improving Deep Neural Networks".

Training neural network requires specifying an initial value of the weights. A well chosen initialization method will help learning to converge faster.  

Previously we didn't talk how the weight initialization works but it has worked out so far. Now let's bring this to the table how do we choose the initialisation for a new neural network.
**In this notebook: **
- We will see how different initialisations lead to different results.

A well chosen initialisation can:
- Speed up the convergence of gradient descent
- Increase the odds of gradient descent converging to a lower training (and generalization) error

To get started, let import the library as well as dataset that we are going to use.


```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()
```


![png](/images/Initialisation/output_1_0.png)


**Problem statement:** We want a classifier to separate the blue dots from the red dots.

### 1 - Neural Network model

I will use a 3-layer neural network which has been implemented below. Here are the initialisation methods I will experiment with:  
- *Zeros initialization* --  setting `initialization = "zeros"` in the input argument.
- *Random initialization* -- setting `initialization = "random"` in the input argument. This initializes the weights to large random values.  
- *He initialization* -- setting `initialization = "he"` in the input argument. This initializes the weights to random values scaled according to a paper by He et al., 2015.


```python
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialisation = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")

    Returns:
    parameters -- parameters learnt by the model
    """

    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]

    # Initialize parameters dictionary.
    if initialisation == "zeros":
        parameters = initialise_parameters_zeros(layers_dims)
    elif initialisation == "random":
        parameters = initialise_parameters_random(layers_dims)
    elif initialisation == "he":
        parameters = initialise_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)

        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
```

### 2 - Zero initialisation

There are two types of parameters to initialise in a neural network:
- the weight matrices $(W^{[1]}, W^{[2]}, W^{[3]}, ..., W^{[L-1]}, W^{[L]})$
- the bias vectors $(b^{[1]}, b^{[2]}, b^{[3]}, ..., b^{[L-1]}, b^{[L]})$

Let's implement the following function to initialise all parameters to zeros. We'll see later that this does not work well since it fails to "break symmetry", but lets try it anyway and see what happens. Use np.zeros((..,..)) with the correct shapes.


```python
def initialise_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    parameters = {}
    L = len(layers_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))

    return parameters
```


```python
parameters = initialise_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

    W1 = [[ 0.  0.  0.]
     [ 0.  0.  0.]]
    b1 = [[ 0.]
     [ 0.]]
    W2 = [[ 0.  0.]]
    b2 = [[ 0.]]


Run the following code to train our model on 15,000 iterations using zeros initialisation.


```python
parameters = model(train_X, train_Y, initialisation = "zeros")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

    Cost after iteration 0: 0.6931471805599453
    Cost after iteration 1000: 0.6931471805599453
    Cost after iteration 2000: 0.6931471805599453
    Cost after iteration 3000: 0.6931471805599453
    Cost after iteration 4000: 0.6931471805599453
    Cost after iteration 5000: 0.6931471805599453
    Cost after iteration 6000: 0.6931471805599453
    Cost after iteration 7000: 0.6931471805599453
    Cost after iteration 8000: 0.6931471805599453
    Cost after iteration 9000: 0.6931471805599453
    Cost after iteration 10000: 0.6931471805599455
    Cost after iteration 11000: 0.6931471805599453
    Cost after iteration 12000: 0.6931471805599453
    Cost after iteration 13000: 0.6931471805599453
    Cost after iteration 14000: 0.6931471805599453



![png](/images/Initialisation/output_10_1.png)


    On the train set:
    Accuracy: 0.5
    On the test set:
    Accuracy: 0.5


The performance is really bad, and the cost does not really decrease, and the algorithm performs no better than random guessing. Why? Lets look at the details of the predictions and the decision boundary:


```python
print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))
```

    predictions_train = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0]]
    predictions_test = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]



```python
plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```


![png](/images/Initialisation/output_13_0.png)


The model is predicting 0 for every example.

In general, initialising all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with $n^{[l]}=1$ for every layer, and the network is no more powerful than a linear classifier such as logistic regression.

**Things to remember here are**:
- The weights $W^{[l]}$ should be initialised randomly to break symmetry.
- It is however okay to initialise the biases $b^{[l]}$ to zeros. Symmetry is still broken so long as $W^{[l]}$ is initialised randomly.


### 3 - Random initialisation

To break symmetry, let's intialise the weights randomly. Following random initialisation, each neuron can then proceed to learn a different function of its inputs. In this exercise, we will see what happens if the weights are intialised randomly but to very large values.

Let's implement the following function to initialise our weights to large random values (scaled by \*10) and our biases to zeros. Use `np.random.randn(..,..) * 10` for weights and `np.zeros((.., ..))` for biases. We are using a fixed `np.random.seed(..)` to make sure our "random" weights will be the same for reproducible purposes.


```python
def initialise_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))

    return parameters
```


```python
parameters = initialise_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

    W1 = [[ 17.88628473   4.36509851   0.96497468]
     [-18.63492703  -2.77388203  -3.54758979]]
    b1 = [[ 0.]
     [ 0.]]
    W2 = [[-0.82741481 -6.27000677]]
    b2 = [[ 0.]]


Run the following code to train our model on 15,000 iterations using random initialisation.


```python
parameters = model(train_X, train_Y, initialisation = "random")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

    Cost after iteration 0: inf


    /home/jovyan/work/week5/Initialization/init_utils.py:145: RuntimeWarning: divide by zero encountered in log
      logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    /home/jovyan/work/week5/Initialization/init_utils.py:145: RuntimeWarning: invalid value encountered in multiply
      logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)


    Cost after iteration 1000: 0.6242434241539614
    Cost after iteration 2000: 0.5978811277755388
    Cost after iteration 3000: 0.5636242569764779
    Cost after iteration 4000: 0.5500958254523324
    Cost after iteration 5000: 0.544339206192789
    Cost after iteration 6000: 0.5373584514307651
    Cost after iteration 7000: 0.469574666760224
    Cost after iteration 8000: 0.39766324943219844
    Cost after iteration 9000: 0.3934423376823982
    Cost after iteration 10000: 0.3920158992175907
    Cost after iteration 11000: 0.38913979237487845
    Cost after iteration 12000: 0.3861261344766218
    Cost after iteration 13000: 0.3849694511273874
    Cost after iteration 14000: 0.3827489017191917



![png](/images/Initialisation/output_20_3.png)


    On the train set:
    Accuracy: 0.83
    On the test set:
    Accuracy: 0.86


If you see "inf" as the cost after the iteration 0, this is because of numerical roundoff; a more numerically sophisticated implementation would fix this. But this isn't worth worrying about for our purposes.

Anyway, it looks like we have broken symmetry, and this gives better results. than before. The model is no longer outputting all 0s.


```python
print (predictions_train)
print (predictions_test)
```

    [[1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 1 1 1 1 1 0 1 1 0 0 1 1
      1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 1 1 0 0 0
      0 0 1 0 1 0 1 1 1 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1
      1 0 0 1 0 0 1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 0 0 0 1 0
      1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 0 1
      0 1 1 1 1 0 1 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 0 1 0 1 1 0 1 1
      0 1 1 0 1 1 1 0 1 1 1 1 0 1 0 0 1 1 0 1 1 1 0 0 0 1 1 0 1 1 1 1 0 1 1 0 1
      1 1 0 0 1 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1
      1 1 1 0]]
    [[1 1 1 1 0 1 0 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0 0 1 0
      1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1
      1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0]]



```python
plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```


![png](/images/Initialisation/output_23_0.png)


**Observations**:
- The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity.
- Poor initialisation can lead to vanishing/exploding gradients, which also slows down the optimisation algorithm.
- If we train this network longer we will see better results, but initialising with overly large random numbers slows down the optimisation.

**In summary**:
- Initialising weights to very large random values does not work well.
- Hopefully intialising with small random values does better. The important question is: how small should be these random values be? Lets find out in the next part!

### 4 - He initialisation

Finally, try "He Initialisation"; this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights $W^{[l]}$ of `sqrt(1./layers_dims[l-1])` where He initialisation would use `sqrt(2./layers_dims[l-1])`.)

Let's implement the following function to initialise your parameters with He initialisation.

**Tricks:** This function is similar to the previous `initialise_parameters_random(...)`. The only difference is that instead of multiplying `np.random.randn(..,..)` by 10, I will multiply it by $\sqrt{\frac{2}{\text{dimension of the previous layer}}}$, which is what He initialisation recommends for layers with a ReLU activation.


```python
def initialise_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))

    return parameters
```


```python
parameters = initialise_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

    W1 = [[ 1.78862847  0.43650985]
     [ 0.09649747 -1.8634927 ]
     [-0.2773882  -0.35475898]
     [-0.08274148 -0.62700068]]
    b1 = [[ 0.]
     [ 0.]
     [ 0.]
     [ 0.]]
    W2 = [[-0.03098412 -0.33744411 -0.92904268  0.62552248]]
    b2 = [[ 0.]]


Run the following code to train our model on 15,000 iterations using He initialisation.


```python
parameters = model(train_X, train_Y, initialisation = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

    Cost after iteration 0: 0.8830537463419761
    Cost after iteration 1000: 0.6879825919728063
    Cost after iteration 2000: 0.6751286264523371
    Cost after iteration 3000: 0.6526117768893807
    Cost after iteration 4000: 0.6082958970572938
    Cost after iteration 5000: 0.5304944491717495
    Cost after iteration 6000: 0.4138645817071794
    Cost after iteration 7000: 0.3117803464844441
    Cost after iteration 8000: 0.23696215330322562
    Cost after iteration 9000: 0.18597287209206834
    Cost after iteration 10000: 0.15015556280371806
    Cost after iteration 11000: 0.12325079292273546
    Cost after iteration 12000: 0.09917746546525934
    Cost after iteration 13000: 0.08457055954024278
    Cost after iteration 14000: 0.07357895962677369



![png](/images/Initialisation/output_29_1.png)


    On the train set:
    Accuracy: 0.993333333333
    On the test set:
    Accuracy: 0.96



```python
plt.title("Model with He initialisation")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```


![png](/images/Initialisation/output_30_0.png)


**Observations**:
- The model with He initialisation separates the blue and the red dots very well in a small number of iterations.


### 5 - Conclusions

We have seen three different types of initialisations. For the same number of iterations and same hyperparameters the comparison is:

<table>
    <tr>
        <td>
        **Model**
        </td>
        <td>
        **Train accuracy**
        </td>
        <td>
        **Problem/Comment**
        </td>

    </tr>
        <td>
        3-layer NN with zeros initialisation
        </td>
        <td>
        50%
        </td>
        <td>
        fails to break symmetry
        </td>
    <tr>
        <td>
        3-layer NN with large random initialisation
        </td>
        <td>
        83%
        </td>
        <td>
        too large weights
        </td>
    </tr>
    <tr>
        <td>
        3-layer NN with He initialisation
        </td>
        <td>
        99%
        </td>
        <td>
        recommended method
        </td>
    </tr>
</table>

**The key points for this notebook are**:
- Different initialisations lead to different results
- Random initialisation is used to break symmetry and make sure different hidden units can learn different things
- Don't intialise to values that are too large
- He initialisation works well for networks with ReLU activations.

*last edited: 31/05/19*

<a href="#top">Go to top</a>
