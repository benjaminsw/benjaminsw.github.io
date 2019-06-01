---
permalink: /Multivariate-Regression/
header:
  image: "/images/digital-transition2.jpg"
---

### Multiple Regression (Gradient Descent)

In this notebook we will apply multivariate to estimate multiple regression weights via gradient descent.

* To implement I will add a constant column of 1's to a graphlab SFrame to account for the intercept
* Convert an SFrame into a Numpy array
* Write a predict_output() function using Numpy
* Write a numpy function to compute the derivative of the regression weights with respect to a single feature
* Write gradient descent function to compute the regression weights given an initial weight vector, step size and tolerance.
* Use the gradient descent function to estimate regression weights for multiple features

### Import library


```python
%matplotlib inline
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from math import sqrt
```

### Load in house sales data

Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


```python
data = pd.read_csv("kc_house_data.csv")
colname_lst = list(data.columns.values)
coltype_lst =  [str, str, float, float, float, float, int, str, int, int, int, int, int, int, int, int, str, float, float, float, float]
col_type_dict = dict(zip(colname_lst, coltype_lst))
```

### Split data into training data and test data

From the entire dataset above, we spit data into training and test set using numpy.


```python
idx = np.random.rand(len(data)) < 0.8
train = data[idx]
test = data[~idx]
```


```python
#inspect first five elements in the training set
train.head(5)
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
      <td>1.0</td>
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
      <th>2</th>
      <td>5631500400</td>
      <td>20150225T000000</td>
      <td>180000</td>
      <td>2</td>
      <td>1.0</td>
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
      <td>3.0</td>
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
      <td>2.0</td>
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
    <tr>
      <th>5</th>
      <td>7237550310</td>
      <td>20140512T000000</td>
      <td>1225000</td>
      <td>4</td>
      <td>4.5</td>
      <td>5420</td>
      <td>101930</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>11</td>
      <td>3890</td>
      <td>1530</td>
      <td>2001</td>
      <td>0</td>
      <td>98053</td>
      <td>47.6561</td>
      <td>-122.005</td>
      <td>4760</td>
      <td>101930</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
#inspect first five elements in the test set
test.head(5)
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
      <th>1</th>
      <td>6414100192</td>
      <td>20141209T000000</td>
      <td>538000</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
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
      <th>6</th>
      <td>1321400060</td>
      <td>20140627T000000</td>
      <td>257500</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1715</td>
      <td>0</td>
      <td>1995</td>
      <td>0</td>
      <td>98003</td>
      <td>47.3097</td>
      <td>-122.327</td>
      <td>2238</td>
      <td>6819</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1875500060</td>
      <td>20140731T000000</td>
      <td>395000</td>
      <td>3</td>
      <td>2.00</td>
      <td>1890</td>
      <td>14040</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1890</td>
      <td>0</td>
      <td>1994</td>
      <td>0</td>
      <td>98019</td>
      <td>47.7277</td>
      <td>-121.962</td>
      <td>1890</td>
      <td>14018</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2524049179</td>
      <td>20140826T000000</td>
      <td>2000000</td>
      <td>3</td>
      <td>2.75</td>
      <td>3050</td>
      <td>44867</td>
      <td>1.0</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>9</td>
      <td>2330</td>
      <td>720</td>
      <td>1968</td>
      <td>0</td>
      <td>98040</td>
      <td>47.5316</td>
      <td>-122.233</td>
      <td>4110</td>
      <td>20336</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1202000200</td>
      <td>20141103T000000</td>
      <td>233000</td>
      <td>3</td>
      <td>2.00</td>
      <td>1710</td>
      <td>4697</td>
      <td>1.5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>1710</td>
      <td>0</td>
      <td>1941</td>
      <td>0</td>
      <td>98002</td>
      <td>47.3048</td>
      <td>-122.218</td>
      <td>1030</td>
      <td>4705</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Data visualisation

In ths section, we will look at some of the data visually in order to get the gist of data


```python
# let inspect the distribution of 'price'
sns.distplot(train["price"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17635ac8>




![png](/images/Multivariate-Regression/output_12_1.png)


The following will plot longitude&latitude of houses in this data set to see if they are all located in  Seattle, WA and aslo gives me an idea wheresabout Seattle is located.


```python
# Create a figure of size (i.e. pretty big)
fig = plt.figure(figsize=(20,10))

# Create a map, using the Gall–Peters projection,
map = Basemap(projection='gall',
              # with low resolution,
              resolution = 'l',
              # And threshold 100000
              area_thresh = 100000.0,
              # Centered at 0,0 (i.e null island)
              lat_0=0, lon_0=0)

# Draw the coastlines on the map
map.drawcoastlines()

# Draw country borders on the map
map.drawcountries()

# Fill the land with grey
map.fillcontinents(color = '#888888')

# Draw the map boundaries
map.drawmapboundary(fill_color='#f4f4f4')

# Define our longitude and latitude points
# We have to use .values because of a wierd bug when passing pandas data
# to basemap.
x,y = map(train['long'].values, train['lat'].values)

# Plot them using round markers of size 6
map.plot(x, y, 'ro', markersize=6)

# Show the map
plt.show()
```


![png](/images/Multivariate-Regression/output_14_0.png)


# Convert to Numpy Array

Now we will write a function that will accept a dataframe, a list of feature names (e.g. ['sqft_living', 'bedrooms']) and an target feature e.g. ('price') and will return two things:
* A numpy matrix whose columns are the desired features plus a constant column (this is how we create an 'intercept')
* A numpy array containing the values of the output


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

For testing let's use the 'sqft_living' feature and a constant as our features and price as our output:


```python
(example_features, example_output) = get_numpy_data(train, ['sqft_living'], 'price')
print example_features[0,:] # this accesses the first row of the data the ':' indicates 'all columns'
print example_output[0] # and the corresponding output
```

    [   1 1180]
    221900.0


# Predicting output given regression weights

Suppose we had the weights [1.0, 1.0] and the features [1.0, 1180.0] and we wanted to compute the predicted output 1.0\*1.0 + 1.0\*1180.0 = 1181.0 this is the dot product between these two arrays. If they're numpy arrays we can use np.dot() to compute this:


```python
my_weights = np.array([1., 1.]) # the example weights
my_features = example_features[0,] # we'll use the first data point
predicted_value = np.dot(my_features, my_weights)
print predicted_value
```

    1181.0


np.dot() also works when dealing with a matrix and a vector. The predictions from all the observations is the dot product between the features *matrix* and the weights *vector*. The following predict_output function is to compute the predictions for an entire matrix of features given the matrix and the weights:


```python
def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
```

The following cell is used to test the code:


```python
test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0
```

    1181.0
    771.0


# Computing the Derivative

We are now going to compute the derivative of the regression cost function. The cost function is the sum over the data points of the squared difference between an observed output and a predicted output.

Since the derivative of a sum is the sum of the derivatives we can compute the derivative for a single data point and then sum over data points. We can write the squared difference between the observed output and predicted output for a single point as follows:

$$(w_0 \cdot h_0(x_i)+(w_1 \cdot h_1(x_i) + \dots+w_k \cdot h_k(w_k)-y_i)^2 = w[0]\cdot[CONSTANT] + w[1]\cdot[feature_1] +\dots +  w[k]\cdot[feature_k] - output)^2 $$

Where we have k features and a constant. So the derivative **with respect to weight w[i]** by the chain rule is:

$$2 \cdot (w[0]\cdot [CONSTANT] + w[1]\cdot [feature_1] + \dots + w[i] \cdot [feature_i] + \dots +  w[k]\cdot [feature_k] - output)\cdot [feature_i]$$

The term inside the paranethesis is just the error (difference between prediction and output). So we can re-write this as:

$$2\cdot error[feature_i]$$

That is, the derivative for the weight for feature i is the sum (over data points) of 2 times the product of the error and the feature itself.
In the case of the constant then this is just twice the sum of the errors as thr feature is constant and represented by array of 1s.

Note: twice the sum of the product of two vectors is just twice the dot product of the two vectors. Therefore the derivative for the weight for feature_i is just two times the dot product between the values of feature_i and the current errors.

The following derivative function computes the derivative of the weight given the value of the feature (over all data points) and the errors (over all data points).


```python
def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2*np.dot(errors, feature)

    return(derivative)
```

To test your feature derivartive run the following:


```python
(example_features, example_output) = get_numpy_data(train, ['sqft_living'], 'price')
my_weights = np.array([0., 0.]) # this makes all the predictions 0
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors in this case is just the -example_output
feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
derivative = feature_derivative(errors, feature)
print derivative
print -np.sum(example_output)*2 # should be the same as derivative
```

    -18686397554.0
    -18686397554.0


# Gradient Descent

Now we will write a function that performs a gradient descent. The basic premise is simple. Given a starting point we update the current weights by moving in the negative gradient direction. Note that the gradient is the direction of *increase* and therefore the negative gradient is the direction of *decrease* and we're trying to *minimise* a cost/error function.

The amount by which we move in the negative gradient *direction*  is called the 'step size'. We stop when we are 'sufficiently close' to the optimum. We define this by requiring that the magnitude of the gradient vector to be smaller than a fixed 'tolerance'.

The following gradient descent function uses the derivative function above. For each step in the gradient descent we update the weight for each feature befofe computing our stopping criteria


```python
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights) # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        error = predictions - output
        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            feature = feature_matrix[:, i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(error,feature)
            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += derivative ** 2
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - np.dot(step_size,derivative)
        # compute the square-root of the gradient sum of squares to get the gradient matnigude/length of a vector:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
        #print gradient_magnitude
        #time.sleep(1)
    return(weights)
```

A few things to note before running the gradient descent. Since the gradient is a sum over all the data points and involves a product of an error and a feature the gradient itself will be very large since the features are large (squarefeet) and the output is large (prices). So while we expect "tolerance" to be small, small is only relative to the size of the features.

For similar reasons the step size will be much smaller than we might expect but this is because the gradient has such large values.

# Running the Gradient Descent as Simple Regression

First let's split the data into training and test data.

Although the gradient descent is designed for multiple regression since the constant is now a feature we can use the gradient descent function to estimat the parameters in the simple regression on squarefeet. The folowing cell sets up the feature_matrix, output, initial weights and step size for the first model:


```python
# let's test out the gradient descent
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train, simple_features, my_output)
initial_weights = np.array([1., 1.])
step_size = 7e-12
tolerance = 2.5e8
```

Next let's run our gradient descent with the above parameters.


```python
model_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
model_weights
```




    array([-1230.97379315,   263.52246994])



Use our newly estimated weights and your predict_output() function to compute the predictions on all the TEST data using test_simple_feature_matrix and your weights from above.


```python
(test_simple_feature_matrix, test_output) = get_numpy_data(test, simple_features, my_output)
simple_weights = regression_gradient_descent(test_simple_feature_matrix, test_output, initial_weights, step_size, tolerance)
predictions = predict_output(test_simple_feature_matrix, simple_weights)
```

Now that we have the predictions on test data, let's compute the RSS on the test data set.


```python
def compute_rss(test_output, predictions):
    residuals = test_output - predictions
    rss = (residuals * residuals).sum()
    return rss
compute_rss(test_output, predictions)
```




    292435195058503.87



# Running a multiple regression

Now we will use more than one actual feature. Use the following code to produce the weights for a second model with the following parameters:


```python
model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors.
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train, model_features, my_output)
initial_weights = np.array([1., 1., 1.])
step_size = 4e-12
tolerance = 3e9
```

Now, we will use the above parameters to estimate the model weights.


```python
model_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
model_weights
```




    array([   0.66082694,  240.70210762,   25.0654162 ])



Use our newly estimated weights and the predict_output function to compute the predictions on the TEST data.


```python
(test_simple_feature_matrix, test_output) = get_numpy_data(test, model_features, my_output)
simple_weights = regression_gradient_descent(test_simple_feature_matrix, test_output, initial_weights, step_size, tolerance)
predictions = predict_output(test_simple_feature_matrix, simple_weights)
```

Now that we have the predictions on test data, let's compute the RSS on the test data set.


```python
def compute_rss(test_output, predictions):
    residuals = test_output - predictions
    rss = (residuals * residuals).sum()
    return rss
compute_rss(test_output, predictions)
```




    291333833505609.0



*last edit 25/10/2016*
