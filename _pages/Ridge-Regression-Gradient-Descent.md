---
permalink: /Ridge-Regression-Gradient-Descent/
header:
  image: "/images/digital-transition2.jpg"
---

### Ridge Regression Implementation with Gradient Descent

In this notebook, we will implement ridge regression via gradient descent.

**outline for this notebook**

* We will write a Numpy function to compute the derivative of the regression weights with respect to a single feature
* We will write gradient descent function to compute the regression weights given an initial weight vector, step size, tolerance, and L2 penalty


### import library

First of all, let's import lirary that we will use in this notebook.


```python
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
```

### read house sales data in

Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


```python
df = pd.read_csv("kc_house_data.csv")
colname_lst = list(df.columns.values)
coltype_lst =  [str, str, float, float, float, float, int, str, int, int, int, int, int, int, int, int, str, float, float, float, float]
col_type_dict = dict(zip(colname_lst, coltype_lst))
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>20141013T000000</td>
      <td>221900</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>20141209T000000</td>
      <td>538000</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>20150225T000000</td>
      <td>180000</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>20141209T000000</td>
      <td>604000</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>20150218T000000</td>
      <td>510000</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### useful functions for later use

Now we will write a function to convert our dataframe to numpy array.


```python
def get_numpy_data(df, features, output):
    df['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_df = df[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_df.as_matrix()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_serie = df[output]
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_serie.as_matrix()
    return(feature_matrix, output_array)
```

We also need the `predict_output()` function to compute the predictions for an entire matrix of features given the matrix and the weights. This function is defined as below.


```python
def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
```

### computing the derivative

We are now going to move to computing the derivative of the regression cost function. First of all, the cost function is the sum over the data points of the squared difference between an observed output and a predicted output, plus the L2 penalty term.
```
Cost(w)
= SUM[ (prediction - output)^2 ]
+ l2_penalty*(w[0]^2 + w[1]^2 + ... + w[k]^2).
```

Since the derivative of a sum is the sum of the derivatives, we can take the derivative of the first part (the RSS) and add the derivative of the regularization part.  As we saw, the derivative of the RSS with respect to `w[i]` can be written as:
```
2*SUM[ error*[feature_i] ].
```
The derivative of the regularization term with respect to `w[i]` is:
```
2*l2_penalty*w[i].
```
Summing both terms, we get
```
2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i].
```
That is, the derivative for the weight for feature i is the sum (over data points) of 2 times the product of the error and the feature itself, plus `2*l2_penalty*w[i]`.

**We will not regularize the constant.**  Thus, in the case of the constant, the derivative is just twice the sum of the errors (without the `2*l2_penalty*w[0]` term).

Now that twice the sum of the product of two vectors is just twice the dot product of the two vectors, therefore the derivative for the weight for feature_i is just two times the dot product between the values of feature_i and the current errors, plus `2*l2_penalty*w[i]`.

With this in mind complete the following derivative function which computes the derivative of the weight given the value of the feature (over all data points) and the errors (over all data points).  To decide when to we are dealing with the constant (so we don't regularize it) we added the extra parameter to the call `feature_is_constant` which you should set to `True` when computing the derivative of the constant and `False` otherwise.


```python
#  get derivative for a particular weight i
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    derivative = 2 * np.dot(errors, feature)
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if not feature_is_constant:
        derivative +=  2 * l2_penalty * weight
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    return derivative
```

To test your feature derivartive run the following:


```python
(example_features, example_output) = get_numpy_data(df, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20.
print ''

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2.
```

    -5.6554166816e+13
    -5.6554166816e+13

    -22446749330.0
    -22446749330.0


### gradient descent

Now we will write a function that performs a gradient descent. The basic premise is simple. Given a starting point we update the current weights by moving in the negative gradient direction. Recall that the gradient is the direction of *increase* and therefore the negative gradient is the direction of *decrease* and we're trying to *minimize* a cost function.

The amount by which we move in the negative gradient *direction*  is called the 'step size'. We stop when we are 'sufficiently close' to the optimum. We will set a **maximum number of iterations** and take gradient steps until we reach this maximum number. If no maximum number is supplied, the maximum should be set 100 by default. (Use default parameter values in Python.)

With this in mind, the gradient descent function below uses our derivative function above. For each step in the gradient descent, we update the weight for each feature before computing our stopping criteria.


```python
def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights) # make sure it's a numpy array
    iter = 0
    while iter < max_iterations:
    #while not reached maximum number of iterations:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        prediction = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        error = prediction - output
        for i in xrange(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            feature = feature_matrix[:, i]
            #(Remember: when i=0, you are computing the derivative of the constant!)
            derivative = feature_derivative_ridge(error, feature, weights[i], l2_penalty, i == 0)
            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size * derivative
        iter += 1
    return weights
```

### visualising effect of L2 penalty

The L2 penalty gets its name because it causes weights to have small L2 norms than otherwise. Let's see how large weights get penalized. Let us consider a simple model with 1 feature:


```python
simple_features = ['sqft_living']
my_output = 'price'
```

Let us split the dataset into training set and test set. Make sure to use `seed=0`:


```python
idx = np.random.rand(len(df)) < 0.8
train = df[idx]; test = df[~idx]
```

In this part, we will only use `'sqft_living'` to predict `'price'`. Use the `get_numpy_data` function to get a Numpy versions of your data with only this feature, for both the `train_data` and the `test_data`.


```python
(simple_feature_matrix, output) = get_numpy_data(train, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test, simple_features, my_output)
```

Let's set the parameters for our optimization:


```python
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000
```

First, let's consider no regularization.  Set the `l2_penalty` to `0.0` and run our ridge regression algorithm to learn the weights of our model. We'll use them later.


```python
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 0.0, max_iterations)
print simple_weights_0_penalty
```

    [ -1.34976903e-01   2.63316982e+02]


Next, let's consider high regularization. Let's set the `l2_penalty` to `1e11` and run our ridge regression algorithm to learn the weights of our model. Then again, we'll use them later.


```python
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)
print simple_weights_high_penalty
```

    [   9.77876417  124.14708828]


This code will plot the two learned models.  (The blue line is for the model with no regularization and the red line is for the one with high regularization.)


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(simple_feature_matrix,output,'b.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'c-',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')
```




    [<matplotlib.lines.Line2D at 0x1835d5c0>,
     <matplotlib.lines.Line2D at 0x1835d7b8>,
     <matplotlib.lines.Line2D at 0x1835d978>,
     <matplotlib.lines.Line2D at 0x183b7320>,
     <matplotlib.lines.Line2D at 0x183b74a8>,
     <matplotlib.lines.Line2D at 0x183b7e10>]




![png](/images/Ridge-Regression-Gradient-Descent/output_34_1.png)


Now let's compute the RSS on the TEST data for the following three sets of weights:
1. The initial weights (all zeros)
2. The weights learned with no regularization
3. The weights learned with high regularization

This is to see which weights perform best


```python
def get_residual_sum_of_squares(feature_matrix, outcome, weights):
    # First get the predictions
    predicted_price = predict_output(feature_matrix, weights)
    # Then compute the residuals/errors
    residuals = predicted_price - outcome
    # print residuals
    # Then square and add them up
    RSS = (residuals * residuals).sum()
    return(RSS)
```


```python
rss_initial = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, initial_weights)
print "RSS weights learned with initial weights (all zeros): $%.6f" % (rss_initial)
```

    RSS weights learned with initial weights (all zeros): $1842274930383324.000000



```python
rss_weights_noregulalisation = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, simple_weights_0_penalty)
print "RSS weights learned with no regularization: $%.6f" % (rss_weights_noregulalisation)
```

    RSS weights learned with no regularization: $294236342252336.750000



```python
rss_weights_regulalisation = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, simple_weights_high_penalty)
print "RSS weights learned with high regularization: $%.6f" % (rss_weights_regulalisation)
```

    RSS weights learned with high regularization: $723302146993537.375000


### running a multiple regression with L2 penalty

Let us now consider a model with 2 features: `['sqft_living', 'sqft_living15']`.

First, let's create Numpy versions of your training and test data with these two features.


```python
model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors.
my_output = 'price'
(feature_matrix, output) = get_numpy_data(df, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test, model_features, my_output)
```

We need to re-inialise the weights since we have one extra parameter. Let's also set the step size and maximum number of iterations.


```python
initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000
```

First, let's consider no regularisation l2 regression. First of all, set the `l2_penalty` to `0.0` and run your ridge regression algorithm to learn the weights of your model.


```python
feature_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 0.0, max_iterations)
print feature_weights_0_penalty
```

    [  -0.43827092  240.22388014   25.64948575]


Next, let's consider high regularisation. Set the `l2_penalty` to `1e11` and run your ridge regression algorithm to learn the weights of your model.  


```python
feature_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)
print feature_weights_high_penalty
```

    [  7.21126343  99.1176732   84.42549764]


Let's compute the RSS on the TEST data for the following three sets of weights:
1. The initial weights (all zeros)
2. The weights learned with no regularization
3. The weights learned with high regularization

to see which weights perform best


```python
rss_initial_test = get_residual_sum_of_squares(test_feature_matrix, test_output, initial_weights)
print "RSS weights learned with initial weights (all zeros): $%f" % (rss_initial_test)
```

    RSS weights learned with initial weights (all zeros): $1842274930383324.000000



```python
rss_noregularisation_test = get_residual_sum_of_squares(test_feature_matrix, test_output, feature_weights_0_penalty)
print "RSS weights learned with no regularisation: $%f" % (rss_noregularisation_test)
```

    RSS weights learned with no regularisation: $293667115622512.375000



```python
rss_regularisation_test = get_residual_sum_of_squares(test_feature_matrix, test_output, feature_weights_high_penalty)
print "RSS weights learned with high regularisation: $%f" % (rss_regularisation_test)
```

    RSS weights learned with high regularisation: $472597125563755.000000


Now, let's predict the house price for the 1st house in the test set using the no regularisation and high regularisation models.


```python
print 'true house1 price', test_output[0]
```

    true house1 price 510000.0



```python
predicted_price_house1_noregularisation = predict_output(test_feature_matrix, feature_weights_0_penalty)[0]
print "predict the house1 price with no regularisation regression: $%f" % predicted_price_house1_noregularisation
```

    predict the house1 price with no regularisation regression: $449744.754710



```python
predicted_price_house1_regularisation = predict_output(test_feature_matrix, feature_weights_high_penalty)[0]
print  "predict the house1 price with no regularisation regression: $%f" % predicted_price_house1_regularisation
```

    predict the house1 price with no regularisation regression: $318490.798004


Now, let's see which model gives the best prediction by looking at the size of error between actual value and predicted error.


```python
print abs(predicted_price_house1_noregularisation - test_output[0])
print abs(predicted_price_house1_regularisation - test_output[0])
```

    60255.2452902
    191509.201996


We can see that for particular house1, regression with no regularisation perform better because it resuts in less error.

*last edited: 31/10/2016*
