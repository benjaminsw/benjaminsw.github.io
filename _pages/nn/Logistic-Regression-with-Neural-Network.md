---
permalink: /nn/Logistic-Regression-with-Neural-Network/
header:
  image: "/images/digital-transition2.jpg"
---
<h2 id="top"></h2>
### Logistic Regression with a Neural Network

This notebook will build a logistic regression classifier to recognize  cats. This notebook will step through how to do this with a Neural Network.

**In this notebook:**
- I will tend not to use loops either for nor while in my code unless where it's necessary.

**Goals of this Notebook are to:**
- Build the general architecture of a learning algorithm, including:
    - Initializing parameters
    - Calculating the cost function and its gradient
    - Using an optimization algorithm (gradient descent)
- Gather all three functions above into a main model function, in the right order.

### 1 - Packages ##

First, let's run the cell below to import all the packages that you will need during this assignment.
- [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
- [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
- [matplotlib](http://matplotlib.org) is a famous library to plot graphs in Python.
- [PIL](http://www.pythonware.com/products/pil/) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.


```python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline
```

### 2 - Overview of the Problem set ##

**Problem Statement**: In this notebook, I'll use a dataset ("data.h5") containing:
- A training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- A test set of m_test images labeled as cat or non-cat
- Each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).

I will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.

Let's get more familiar with the dataset. Load the data by running the following code.


```python
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```

I added "_orig" at the end of image datasets (train and test) because I am going to preprocess them. After preprocessing, I will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).

Each line of my train_set_x_orig and test_set_x_orig is an array representing an image. I will visualize an example of data by running the following code. The image can be changed to others by changing the index.


```python
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
```

    y = [1], it's a 'cat' picture.



![png](/images/Logistic-Regression-with-Neural-Network/output_6_1.png)


Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. If we can keep our matrix/vector dimensions straight, we will go a long way toward eliminating many bugs.

Let's find the values for:
- m_train (number of training examples)
- m_test (number of test examples)
- num_px (= height = width of a training image)
Note that `train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3). For instance, `m_train` can be accessed by writing `train_set_x_orig.shape[0]`.


```python
m_train = len(train_set_x_orig)
m_test = len(test_set_x_orig)
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
```

    Number of training examples: m_train = 209
    Number of testing examples: m_test = 50
    Height/Width of each image: num_px = 64
    Each image is of size: (64, 64, 3)
    train_set_x shape: (209, 64, 64, 3)
    train_set_y shape: (1, 209)
    test_set_x shape: (50, 64, 64, 3)
    test_set_y shape: (1, 50)


For convenience, I shall now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px $*$ num_px $*$ 3, 1). After this, my training (and test) dataset is a numpy-array where each column represents a flattened image. There should be m_train (respectively m_test) columns.

Let's reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num\_px $*$ num\_px $*$ 3, 1).

A trick I will use to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use:
```python
X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
```


```python
# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
```

    train_set_x_flatten shape: (12288, 209)
    train_set_y shape: (1, 209)
    test_set_x_flatten shape: (12288, 50)
    test_set_y shape: (1, 50)
    sanity check after reshaping: [17 31 56 22 33]


To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.

One common preprocessing step in machine learning is to center and standardize the dataset, meaning that you will substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).

During the training of your model, you're going to multiply weights and add biases to some initial inputs in order to observe neuron activations. Then you backpropogate with the gradients to train the model. But, it is extremely important for each feature to have a similar range such that our gradients don't explode. You will see that more in detail in later notebook.

Let's standardize our dataset.


```python
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```

**Key points for this section are:**
- To figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
- To reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
- To "standardize" the data

### 3 - General Architecture of the learning algorithm ##

It's time to design a simple algorithm to distinguish cat images from non-cat images.

I will build a Logistic Regression, using a Neural Network mindset. The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**

<img src="/images/Logistic-Regression-with-Neural-Network/LogReg_kiank.png" style="width:650px;height:400px;">

**Mathematical expression of the algorithm**:

For one example $x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

The cost is then computed by summing over all training examples:
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$

**The below cell will implement the following:**
- Initialize the parameters of the model
- Learn the parameters for the model by minimizing the cost  
- Use the learned parameters to make predictions (on the test set)
- Analyse the results and conclude

### 4 - Building the parts of the algorithm ##

The main steps for building a Neural Network are:
1. Define the model structure (such as number of input features)
2. Initialize the model's parameters
3. Loop:
    - Calculate current loss (forward propagation)
    - Calculate current gradient (backward propagation)
    - Update parameters (gradient descent)

I will build 1-3 separately and integrate them into one function and call it `model()`.

### 4.1 - Helper functions

Below, I will implement `sigmoid()` function as seen in the figure above. Sigmoid is defined as $sigmoid( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$ .


```python
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))

    return s
```


```python
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
```

    sigmoid([0, 2]) = [ 0.5         0.88079708]


### 4.2 - Initialising parameters

Next, I will implement parameter initialisation in the cell below. I will initialise w as a vector of zeros. The numpy function to use is np.zeros(). This function will initialise an array of zero's.


```python
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b
```


```python
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
```

    w = [[ 0.]
     [ 0.]]
    b = 0


**Note:** For image inputs, w will be of shape (num_px $\times$ num_px $\times$ 3, 1).

### 4.3 - Forward and Backward propagation

Now that our parameters are initialised, I can do the "forward" and "backward" propagation steps for learning the parameters.

**The below cell will implement the following:**

- Implement a function `propagate()` that computes the cost function and its gradient.

**Tricks:**
Forward Propagation:
- Obtain X
- Compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
- Calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$

Here are the two formulas I will be using:

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$


```python
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(X.T,w)+b)
    # compute activation
    cost = -1/m*(np.sum(Y*np.log(A.T)+(1-Y)*np.log(1-A.T)))             # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m*np.dot(X,(A-Y.T))
    db = 1/m*np.sum(A-Y.T)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost
```


```python
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
```

    dw = [[ 0.99845601]
     [ 2.39507239]]
    db = 0.00145557813678
    cost = 5.80154531939


### 4.4 - Optimisation

Previously, I have initialised the parameters and I am also able to compute a cost function and its gradient. Now, I want to update the parameters using gradient descent.


**The below cell will implement the following:**
- Implement the optimisation function.

**Note:**
- The goal is to learn $w$ and $b$ by minimising the cost function $J$. For a parameter $\theta$, the update rule is $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.


```python
def optimise(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):


        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)


        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
```


```python
params, grads, costs = optimise(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
```

    w = [[ 0.19033591]
     [ 0.12259159]]
    b = 1.92535983008
    dw = [[ 0.67752042]
     [ 1.41625495]]
    db = 0.219194504541


The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X. Implement the `predict()` function. There are two steps to computing predictions.

**The below cell will implement the following:**
- Calculate $\hat{Y} = A = \sigma(w^T X + b)$
- Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`. If you wish, you can use an `if`/`else` statement in a `for` loop (though there is also a way to vectorize this).


```python
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(X.T,w)+b)

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        #Y_prediction = 1. if A < 0.5 else 0.
        Y_prediction = np.where(A > 0.5, 1., 0.).T

    assert(Y_prediction.shape == (1, m))

    return Y_prediction
```


```python
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))
```

    predictions = [[ 1.  1.  0.]]


**What we have done so far:**
I have implemented functions that:
- Initialise (w,b)
- Optimise the loss iteratively to learn parameters (w,b):
    - Computing the cost and its gradient
    - Updating the parameters using gradient descent
- Use the learned (w,b) to predict the labels for a given set of examples

### 5 - Merge all functions into a model ##

We will now put everytghing that we have implemented and see how the overall model is structured by putting together all the building blocks (functions implemented in the previous parts) together, in the right order.

**Notation:**
- Y_prediction_test for our predictions on the test set
- Y_prediction_train for our predictions on the train set
- w, costs, grads for the outputs of optimise()


```python
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = np.zeros((num_px*num_px*3,1)), 0
    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimise(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d
```

Run the following cell to train your model.


```python
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
```

    Cost after iteration 0: 0.693147
    Cost after iteration 100: 0.584508
    Cost after iteration 200: 0.466949
    Cost after iteration 300: 0.376007
    Cost after iteration 400: 0.331463
    Cost after iteration 500: 0.303273
    Cost after iteration 600: 0.279880
    Cost after iteration 700: 0.260042
    Cost after iteration 800: 0.242941
    Cost after iteration 900: 0.228004
    Cost after iteration 1000: 0.214820
    Cost after iteration 1100: 0.203078
    Cost after iteration 1200: 0.192544
    Cost after iteration 1300: 0.183033
    Cost after iteration 1400: 0.174399
    Cost after iteration 1500: 0.166521
    Cost after iteration 1600: 0.159305
    Cost after iteration 1700: 0.152667
    Cost after iteration 1800: 0.146542
    Cost after iteration 1900: 0.140872
    train accuracy: 99.04306220095694 %
    test accuracy: 70.0 %


**Expected Output**:

<table style="width:40%">

    <tr>
        <td> **Cost after iteration 0 **  </td>
        <td> 0.693147 </td>
    </tr>
      <tr>
        <td> <center> $\vdots$ </center> </td>
        <td> <center> $\vdots$ </center> </td>
    </tr>  
    <tr>
        <td> **Train Accuracy**  </td>
        <td> 99.04306220095694 % </td>
    </tr>

    <tr>
        <td>**Test Accuracy** </td>
        <td> 70.0 % </td>
    </tr>
</table>




**Comment**: Training accuracy is close to 100%. This is a good sanity check: our model is working and has high enough capacity to fit the training data. Test error is 68%. It is actually not bad for this simple model, given the small dataset we used and that logistic regression is a linear classifier. But no worries, later on we will build an even better classifier!

Also, we can see that the model is clearly overfitting the training data. Later in this specialization we will get into how to reduce overfitting, for example by using regularization.

Using the code below (and changing the `index` variable) we can look at predictions on pictures of the test set.


```python
# Example of a picture that was wrongly classified.
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
```

    y = 1, you predicted that it is a "cat" picture.



![png](/images/Logistic-Regression-with-Neural-Network/output_38_1.png)


Let's also plot the cost function and the gradients.


```python
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
```


![png](/images/Logistic-Regression-with-Neural-Network/output_40_0.png)


**Interpretation**:
We can see the cost decreasing. It shows that the parameters are being learned. However, we can see that we could train the model even more on the training set. Try to increase the number of iterations in the cell above and rerun the cells. You might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting.

### 6 - Further analysis


#### Choice of learning rate ####

**Reminder**:
In order for Gradient Descent to work we must choose the learning rate wisely. The learning rate $\alpha$ determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

Let's compare the learning curve of our model with several choices of learning rates. Let's see in the cell below.


```python
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```

    learning rate is: 0.01
    [{'w': array([[ 0.01280149],
           [-0.03466593],
           [-0.01552747],
           ...,
           [-0.01481023],
           [-0.03723668],
           [ 0.03117718]]), 'b': -0.0038534039511558212}, {'dw': array([[-0.0002544 ],
           [ 0.00068592],
           [ 0.00029474],
           ...,
           [ 0.00019069],
           [ 0.00052802],
           [-0.00086524]]), 'db': 3.1020562256115292e-06}, [0.69314718055994529, 0.82392086816007459, 0.41894370649462165, 0.61734970643446341, 0.52211576711042629, 0.38770874591037602, 0.23625445651846877, 0.15422213305604038, 0.13532782832676457, 0.12497148001139548, 0.11647833126190324, 0.10919251128431111, 0.10280446418273691, 0.09712981007880353, 0.092043269234435954]]
    train accuracy: 99.52153110047847 %
    test accuracy: 68.0 %

    -------------------------------------------------------

    learning rate is: 0.001
    [{'w': array([[ 0.00388218],
           [-0.00802229],
           [-0.00375977],
           ...,
           [-0.0042936 ],
           [-0.01110565],
           [ 0.00659217]]), 'b': -0.011016013419866308}, {'dw': array([[-0.00140391],
           [ 0.00410247],
           [ 0.00202086],
           ...,
           [ 0.00197498],
           [ 0.00487191],
           [-0.00352957]]), 'db': 0.0041431461114086023}, [0.69314718055994529, 0.59128942600035395, 0.55579611071270885, 0.52897651315623651, 0.50688129174355179, 0.48787986321716587, 0.47110827803124367, 0.45604580969828512, 0.44235022793365292, 0.42978171535077836, 0.41816382093643284, 0.40736174995821905, 0.39726946872697977, 0.38780160722954093, 0.37888813035939578]]
    train accuracy: 88.99521531100478 %
    test accuracy: 64.0 %

    -------------------------------------------------------

    learning rate is: 0.0001
    [{'w': array([[ 0.00090564],
           [-0.00099018],
           [-0.00014868],
           ...,
           [-0.00051939],
           [-0.00179446],
           [ 0.00090753]]), 'b': -0.0023254831079343621}, {'dw': array([[-0.0047644 ],
           [ 0.00656143],
           [ 0.00237252],
           ...,
           [ 0.00388103],
           [ 0.01098681],
           [-0.00501018]]), 'db': 0.010661211290196081}, [0.69314718055994529, 0.64367675569352134, 0.63573718140598356, 0.62857204564926517, 0.62203950102519978, 0.61602937869058749, 0.61045508300635853, 0.60524817260857156, 0.60035419173379434, 0.59572948438122253, 0.59133876639131144, 0.58715327545543017, 0.58314935913572941, 0.57930739397930764, 0.57561095484493152]]
    train accuracy: 68.42105263157895 %
    test accuracy: 36.0 %

    -------------------------------------------------------




![png](/images/Logistic-Regression-with-Neural-Network/output_44_1.png)


**Interpretation**:
- Different learning rates give different costs and thus different predictions results.
- If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost).
- A lower cost doesn't mean a better model.We have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
- In deep learning, I usually recommend to:
    - Choose the learning rate that better minimizes the cost function.
    - If your model overfits, use other techniques to reduce overfitting.


### 7 - Prediction step ##

Now we can use our image and see the output of our model. To do that:
1. Add your image to this Jupyter Notebook's directory, in the "images" folder
2. Change your image's name in the following code
3. Run the code and check if the algorithm is right (1 = cat, 0 = non-cat)!


```python
my_image = "my_image.jpg"   # change this to the name of your image file

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```

**Key points for this section are:**
1. Preprocessing the dataset is important.
2. I implemented each function separately: initialize(), propagate(), optimize(), then combine them into built a model().
3. Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm. You will see more examples of this later in this course!

Finally, we can try different things on this Notebook. Things we can play with include:
- Play with the learning rate and the number of iterations
    - Try different initialization methods and compare the results
    - Test other preprocessings (center the data, or divide each row by its standard deviation)

Bibliography:
- <a href="http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/">http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/</a>
- <a href="https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c">https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c</a>


<a href="#top">Go to top</a>
