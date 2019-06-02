---
permalink: /nn/Basic-Numpy/
header:
  image: "/images/digital-transition2.jpg"
---
<h2 id="top"></h2>

### Python Basics with Numpy

This notebok will show you a quick introduction to Python and Jupyter Notebook.

**In this notebook:**
- I will use Python3.
- I will avoid using for-loops and while-loops as performing vecterisation will give cleaner code and better performance.

**Goals of this Notebook are to:**
- Show how to use Jupyter Notebooks
- Show how to use numpy functions and numpy matrix/vector operations
- Illustrate the concept of "broadcasting"
- Show how to vectorise code

Let's get started!

### About iPython Notebooks ##

Jupyter notebooks are interactive coding environments embedded in a webpage. After writing code, the cell can be run by either pressing "SHIFT"+"ENTER" or by clicking on "Run Cell" (denoted by a play symbol) in the upper bar of the notebook.

Now, let's set test to `"Hello World"` in the cell below to print out "Hello World".


```python
# set "test" to 'Hello World'
test = 'Hello World'
```


```python
print ("test: " + test)
```

    test: Hello World


### 1 - Building basic functions with numpy ##

Numpy is the main package for scientific computing in Python. It is maintained by a large community [www.numpy.org](https://www.numpy.org/). In this notebook, I will illustrate key numpy functions such as np.exp, np.log, and np.reshape. All these numpy functions will more or less be used quite ofter in other notebooks.

### 1.1 - sigmoid function, np.exp() ###

Before using np.exp(), I will use math.exp() to implement the sigmoid function. You will then see why np.exp() is preferable to math.exp().

In the following cell, I will build a function that returns the sigmoid of a real number x by using math.exp(x) for the exponential function.

**Note**:
$sigmoid(x) = \frac{1}{1+e^{-x}}$ is sometimes also known as the logistic function. It is a non-linear function used not only in Machine Learning (Logistic Regression), but also in Deep Learning.

<img src="/images/Basic-Numpy/Sigmoid.png" style="width:500px;height:228px;">

To refer to a function belonging to a specific package, the function could be called it using package_name.function(). Let's try the code below to see an example with math.exp().


```python
import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    s = 1/(1+math.exp(-x))

    return s
```


```python
basic_sigmoid(3)
```




    0.9525741268224334



Actually, the "math" library is rarely used in deep learning because the inputs of the functions are real numbers. In deep learning we mostly use matrices and vectors. This is why numpy is more useful.
### One reason why we use "numpy" instead of "math" in Deep Learning ###
x = [1, 2, 3]
basic_sigmoid(x) # this will give an error when running it because x is a vector.
In fact, if $ x = (x_1, x_2, ..., x_n)$ is a row vector then $np.exp(x)$ will apply the exponential function to every element of x. The output will thus be: $np.exp(x) = (e^{x_1}, e^{x_2}, ..., e^{x_n})$


```python
import numpy as np

# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x)) # result is (exp(1), exp(2), exp(3))
```

    [  2.71828183   7.3890561   20.08553692]


Furthermore, if x is a vector, then a Python operation such as $s = x + 3$ or $s = \frac{1}{x}$ will output s as a vector of the same size as x.


```python
# example of vector operation
x = np.array([1, 2, 3])
print (x + 3)
```

    [4 5 6]


Further info on a numpy function can be found at [the official documentation](https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.exp.html).

Alternatively, you can also create a new cell in the notebook and write `np.exp?` (for example) to get quick access to the documentation.

Now, let's implement the sigmoid function using numpy.

Let's x be either a real number, a vector, or a matrix. The data structures I use in numpy to represent these shapes (vectors, matrices...) are called numpy arrays.
$$ \text{For } x \in \mathbb{R}^n \text{,     } sigmoid(x) = sigmoid\begin{pmatrix}
    x_1  \\
    x_2  \\
    ...  \\
    x_n  \\
\end{pmatrix} = \begin{pmatrix}
    \frac{1}{1+e^{-x_1}}  \\
    \frac{1}{1+e^{-x_2}}  \\
    ...  \\
    \frac{1}{1+e^{-x_n}}  \\
\end{pmatrix}\tag{1} $$


```python
import numpy as np # this means you can access numpy functions by writing np.function() instead of numpy.function()

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """

    s = 1/(1+np.exp(-x))

    return s
```


```python
x = np.array([1, 2, 3])
sigmoid(x)
```




    array([ 0.73105858,  0.88079708,  0.95257413])



### 1.2 - Sigmoid gradient

Let's compute gradients to optimise loss functions using backpropagation. Let's code the first gradient function.

Now, I will implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. The formula is: $$sigmoid\_derivative(x) = \sigma'(x) = \sigma(x) (1 - \sigma(x))\tag{2}$$
this function can often be coded in two steps:
1. Set s to be the sigmoid of x.
2. Compute $\sigma'(x) = s(1-s)$


```python
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- My computed gradient.
    """

    s = sigmoid(x) # let's re-use the sigmoid() function that I implemented previously
    ds = s*(1-s)

    return ds
```


```python
x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
```

    sigmoid_derivative(x) = [ 0.19661193  0.10499359  0.04517666]


### 1.3 - Reshaping arrays ###

Two common numpy functions used in deep learning are [np.shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) and [np.reshape()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html).
- X.shape is used to get the shape (dimension) of a matrix/vector X.
- X.reshape(...) is used to reshape X into some other dimension.

For example, in computer science, an image is represented by a 3D array of shape $(length, height, depth = 3)$. However, when an image is read as the input of an algorithm it is converted to a vector of shape $(length*height*3, 1)$. In other words, the 3D array is "unrolled", or reshaped into a 1D vector.

<img src="images/image2vector_kiank.png" style="width:500px;height:300;">

Now, let's implement `image2vector()` that takes an input of shape (length, height, 3) and returns a vector of shape (length\*height\*3, 1). For example, if you would like to reshape an array v of shape (a, b, c) into a vector of shape (a*b,c) you would do:
``` python
v = v.reshape((v.shape[0]*v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
```
**Note**: The dimensions or quantities of image can be looked up with `image.shape[0]`, etc.


```python
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)a

    return v
```


      File "<ipython-input-11-72ad5b23d437>", line 10
        v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)a
                                                                         ^
    SyntaxError: invalid syntax




```python
# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-12-42db90d784ff> in <module>()
         12         [ 0.34144279,  0.94630077]]])
         13
    ---> 14 print ("image2vector(image) = " + str(image2vector(image)))


    NameError: name 'image2vector' is not defined


### 1.4 - Normalising rows

Another common technique often used in Machine Learning and Deep Learning is "normalisation". It often leads to a better performance because gradient descent converges faster after normalisation. Here, by normalisation I mean changing x to $ \frac{x}{\| x\|} $ (dividing each row vector of x by its norm).

For example, if $$x =
\begin{bmatrix}
    0 & 3 & 4 \\
    2 & 6 & 4 \\
\end{bmatrix}\tag{3}$$ then $$\| x\| = np.linalg.norm(x, axis = 1, keepdims = True) = \begin{bmatrix}
    5 \\
    \sqrt{56} \\
\end{bmatrix}\tag{4} $$and        $$ x\_normalised = \frac{x}{\| x\|} = \begin{bmatrix}
    0 & \frac{3}{5} & \frac{4}{5} \\
    \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
\end{bmatrix}\tag{5}$$ Note that we can divide matrices of different sizes and it works fine: this is called broadcasting and I am going to talk about it more in part 5.


Now let's implement normalizeRows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).


```python
def normaliseRows(x):
    """
    Implement a function that normalises each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalised (by row) numpy matrix. You are allowed to modify x.
    """

    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)

    # Divide x by its norm.
    x = x/x_norm

    return x
```


```python
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normaliseRows(x) = " + str(normaliseRows(x)))
```

**Note**:
In normaliseRows(), if I try to print the shapes of x_norm and x, and then rerun the assessment. You'll find out that they have different shapes. This is normal given that x_norm takes the norm of each row of x. So x_norm has the same number of rows but only 1 column. So how did it work when you divided x by x_norm? This is called broadcasting and I'll talk about it now!

### 1.5 - Broadcasting and the softmax function ####
A very important concept to understand in numpy is "broadcasting". It is very useful for performing mathematical operations between arrays of different shapes. For the full details on broadcasting, you can read the official [broadcasting documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

To gain more understanding of this concept, let's implement a softmax function using numpy to help illustrate this. You can think of softmax as a normalising function used when your algorithm needs to classify two or more classes. There is more to come on softmax in the later notebook. So, please stay tuned.

**The below cell will implement the following:**
- $ \text{for } x \in \mathbb{R}^{1\times n} \text{,     } softmax(x) = softmax(\begin{bmatrix}
    x_1  &&
    x_2 &&
    ...  &&
    x_n  
\end{bmatrix}) = \begin{bmatrix}
     \frac{e^{x_1}}{\sum_{j}e^{x_j}}  &&
    \frac{e^{x_2}}{\sum_{j}e^{x_j}}  &&
    ...  &&
    \frac{e^{x_n}}{\sum_{j}e^{x_j}}
\end{bmatrix} $

- $\text{for a matrix } x \in \mathbb{R}^{m \times n} \text{,  $x_{ij}$ maps to the element in the $i^{th}$ row and $j^{th}$ column of $x$, thus we have: }$  $$softmax(x) = softmax\begin{bmatrix}
    x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
    x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{m1} & x_{m2} & x_{m3} & \dots  & x_{mn}
\end{bmatrix} = \begin{bmatrix}
    \frac{e^{x_{11}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{12}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{13}}}{\sum_{j}e^{x_{1j}}} & \dots  & \frac{e^{x_{1n}}}{\sum_{j}e^{x_{1j}}} \\
    \frac{e^{x_{21}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{22}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{23}}}{\sum_{j}e^{x_{2j}}} & \dots  & \frac{e^{x_{2n}}}{\sum_{j}e^{x_{2j}}} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \frac{e^{x_{m1}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m2}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m3}}}{\sum_{j}e^{x_{mj}}} & \dots  & \frac{e^{x_{mn}}}{\sum_{j}e^{x_{mj}}}
\end{bmatrix} = \begin{pmatrix}
    softmax\text{(first row of x)}  \\
    softmax\text{(second row of x)} \\
    ...  \\
    softmax\text{(last row of x)} \\
\end{pmatrix} $$


```python
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """

    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)

    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum

    return s
```


```python
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
```

**Note**:
- If we print the shapes of x_exp, x_sum and s above and rerun the assessment cell, we will see that x_sum is of shape (2,1) while x_exp and s are of shape (2,5). **x_exp/x_sum** works due to python broadcasting.

Yay! We now have a pretty good understanding of python numpy and have implemented a few useful functions that you will be using in deep learning.

**Key points for this section are:**
- np.exp(x) works for any np.array x and applies the exponential function to every coordinate
- how the sigmoid function and its gradient are implemented
- image2vector is commonly used in deep learning
- np.reshape is widely used as we'll see that keeping our matrix/vector dimensions straight will go toward eliminating a lot of bugs in the future.
- numpy has efficient built-in functions
- broadcasting is extremely useful

### 2) Vectorisation


In deep learning, we will deal with very large datasets. Hence, a non-computationally-optimal function can become a huge bottleneck in our algorithm and can result in a model that takes ages to run. To make sure that our code is  computationally efficient, we will use vectorisation. For example, try to tell the difference between the following implementations of the dot/outer/elementwise product.


```python
import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```


```python
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-13-0d8d8315a1c2> in <module>()
          3
          4 ### VECTORIZED DOT PRODUCT OF VECTORS ###
    ----> 5 tic = time.process_time()
          6 dot = np.dot(x1,x2)
          7 toc = time.process_time()


    NameError: name 'time' is not defined


As you may have noticed, the vectorised implementation is much cleaner and more efficient. For bigger vectors/matrices, the differences in running time become even bigger.

**Note** that `np.dot()` performs a matrix-matrix or matrix-vector multiplication. This is different from `np.multiply()` and the `*` operator (which is equivalent to  `.*` in Matlab/Octave), which performs an element-wise multiplication.

### 2.1 Implement the L1 and L2 loss functions

- The loss is used to evaluate the performance of your model. The bigger your loss is, the more different your predictions ($ \hat{y} $) are from the true values ($y$). In deep learning, we use optimisation algorithms like Gradient Descent to train our model and to minimise the cost.

**Recall:**
- Least absolute deviations(L1) is a loss function used to minimise the absolute differences between the prediction and the actual values.
- L1 loss is defined as:
$$\begin{align*} & L_1(\hat{y}, y) = \sum_{i=0}^m|y^{(i)} - \hat{y}^{(i)}| \end{align*}\tag{6}$$

Now let's implement the numpy vectorised version of the L1 loss in the following cell.


```python
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """

    loss = np.sum(np.abs(yhat - y))

    return loss
```


```python
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
```

**Recall**:
- Least square errors(L2) function minimises the squared differences between the prediction values and actual values
- L2 loss is defined as $$\begin{align*} & L_2(\hat{y},y) = \sum_{i=0}^m(y^{(i)} - \hat{y}^{(i)})^2 \end{align*}\tag{7}$$

Now let's implement the numpy vectorised version of the L2 loss. There are several way of implementing the L2 loss but for this time, I will use np.dot() to implement L2. As a reminder, if $x = [x_1, x_2, ..., x_n]$, then `np.dot(x,x)` = $\sum_{j=0}^n x_j^{2}$.


```python
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """

    y_diff = yhat - y
    loss = np.sum(np.dot(y_diff, y_diff))

    return loss
```


```python
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
```

Yay! Finally we reached the end of this notebook and this little warmup will help us in the future notebook when we do neural networks.

**Key points for this section are:**
- Vectorisation is very important in deep learning. It provides computational efficiency and clarity.
- We now know how to implement the L1 and L2 loss.
- We are familiar with many numpy functions such as np.sum, np.dot, np.multiply, np.maximum, etc...

*last edited: 29/05/19*

<a href="#top">Go to top</a>
