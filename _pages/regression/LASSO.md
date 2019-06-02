---
permalink: /regression/LASSO/
header:
  image: "/images/digital-transition2.jpg"
---
<h2 id="top"></h2>

### LASSO (coordinate descent)

In this notebook,we will implement our LASSO solver via coordinate descent. <br>

** outline **
* We will write a function to normalize features
* We will implement coordinate descent for LASSO
* We will explore effects of L1 penalty

### import library


```python
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
```

### read data in

Dataset used in this notebook is from house sales in King County, the region where the city of Seattle, WA is located.


```python
data = pd.read_csv("kc_house_data.csv")
colname_lst = list(data.columns.values)
coltype_lst =  [str, str, float, float, float, float, int, str, int, int, int, int, int, int, int, int, str, float, float, float, float]
col_type_dict = dict(zip(colname_lst, coltype_lst))
data.head()
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

We also need the predict_output() function to compute the predictions for an entire matrix of features given the matrix and the weights. This function is defined as below.


```python
def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
```

### normalise features
In the house dataset, features vary wildly in their relative magnitude: `sqft_living` is very large overall compared to `bedrooms`, for instance. As a result, weight for `sqft_living` would be much smaller than weight for `bedrooms`. This is problematic because "small" weights are dropped first as `l1_penalty` goes up.

To give equal considerations for all features, we need to **normalise features**. We will divide each feature by its 2-norm so that the transformed feature has norm 1.

$$2-norm = |x| = \sqrt{\sum_{i=1}^{n} | x_i |^2}$$
Let's see how we can do this normalization easily with Numpy: let us first consider a small matrix.


```python
X = np.array([[3.,5.,8.],[4.,12.,15.]])
print X
```

    [[  3.   5.   8.]
     [  4.  12.  15.]]


Numpy provides a shorthand for computing 2-norms of each column:


```python
norms = np.linalg.norm(X, axis=0) # gives [norm(X[:,0]), norm(X[:,1]), norm(X[:,2])]
print norms
```

    [  5.  13.  17.]


To normalise, apply element-wise division:


```python
print X / norms # gives [X[:,0]/norm(X[:,0]), X[:,1]/norm(X[:,1]), X[:,2]/norm(X[:,2])]
```

    [[ 0.6         0.38461538  0.47058824]
     [ 0.8         0.92307692  0.88235294]]


Using the shorthand we just covered, Now we will write a short function called `normalise_features(feature_matrix)`, which normalises columns of a given feature matrix. The function will return a pair `(normalized_features, norms)`, where the second item contains the norms of original features. As discussed, we will use these norms to normalise the test data in the same way as we normalised the training data.


```python
def normalise_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    return (feature_matrix/norms, norms)
```

To test the function, run the following:


```python
features, norms = normalise_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print features
# should print
# [[ 0.6  0.6  0.6]
#  [ 0.8  0.8  0.8]]
print norms
# should print
# [5.  10.  15.]
```

    [[ 0.6  0.6  0.6]
     [ 0.8  0.8  0.8]]
    [  5.  10.  15.]


### implementing coordinate descent with normalised features

We seek to obtain a sparse set of weights by minimising the LASSO cost function


$$ Cost(W) = RSS(W) + \lambda * {(Sum \space of \space Absolute \space Values \space of \space Weights)}$$

$$ = \sum{(\hat{y_i}-y_i)^2} + \lambda * {(|w_1|+\dots+|w_k|)}$$
OR
```
                    SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|).
```

Note that by convention, we do not include `w[0]` in the L1 penalty term because we never want to push the intercept to zero, intentaionally.

The absolute value sign makes the cost function non-differentiable, so simple gradient descent is not viable. Therefore, we would need to implement a method called **subgradient descent**. Instead, we will use **coordinate descent**: at each iteration, we will fix all weights but weight `i` and find the value of weight `i` that minimises the objective. That is, we're looking for
```
argmin_{w[i]} [ SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|) ]
```
where all weights other than `w[i]` are held to be constant. We will optimize one `w[i]` at a time, circling through the weights multiple times.  
  1. Pick a coordinate `i`
  2. Compute `w[i]` that minimises the cost function `SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|)`
  3. Repeat Steps 1 and 2 for all coordinates, multiple times

For this notebook, we use **cyclical coordinate descent with normalised features**, where we cycle through coordinates 0 to (d-1) in order, and assume the features were normalised as discussed above. The formula for optimizing each coordinate is as follows:
```
       ┌ (ro[i] + lambda/2)     if ro[i] < -lambda/2
w[i] = ├ 0                      if -lambda/2 <= ro[i] <= lambda/2
       └ (ro[i] - lambda/2)     if ro[i] > lambda/2
```
where
```
ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].
```

Note that we do not regularise the weight of the constant feature (intercept) `w[0]`, so, for this weight, the update is simply:
```
w[0] = ro[i]
```

### effect of L1 penalty

Now let's consider a simple model with 2 features:


```python
simple_features = ["sqft_living", "bedrooms"]
my_output = "price"
(simple_feature_matrix, output) = get_numpy_data(data, simple_features, my_output)
```

Now, let's normalise features


```python
simple_feature_matrix, norms = normalise_features(simple_feature_matrix)
```

We assign some random set of initial weights and inspect the values of `ro[i]`:


```python
weights = np.array([1., 4., 1.])
```

Use `predict_output()` to make predictions on this data.


```python
prediction = predict_output(simple_feature_matrix, weights)
```

Now, let's compute the values of `ro[i]` for each feature in this simple model, using the formula given above, using the formula:
```
ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
```


```python
def get_ro(simple_feature_matrix, output, weights, i):
    prediction = predict_output(simple_feature_matrix, weights)
    feature_i = simple_feature_matrix[:, i]
    ro_i = (feature_i * (output - prediction + weights[i] * feature_i)).sum()
    return ro_i
```

Note that whenever `ro[i]` falls between `-l1_penalty/2` and `l1_penalty/2`, the corresponding weight `w[i]` is sent to zero.

### single coordinate descent step

Using the formula above, now we will implement coordinate descent that minimises the cost function over a single feature i. Note that the intercept (weight 0) is not regularised. The function should accept feature matrix, output, current weights, l1 penalty, and index of feature to optimise over. The function should return new weight for feature i.


```python
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = get_ro(feature_matrix, output, weights, i)

    if i == 0: # intercept -- not regularised
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2.
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2.
    else:
        new_weight_i = 0.

    return new_weight_i
```

To test the function, run the following cell:


```python
# should print 0.425558846691
import math
print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]),
                                   np.array([1., 1.]), np.array([1., 4.]), 0.1)
```

    0.425558846691


### cyclical coordinate descent

Now that we have a function that optimises the cost function over a single coordinate, let's implement cyclical coordinate descent where we optimise coordinates 0, 1, ..., (d-1) in order and repeat.

**stop criteria** Each time we scan all the coordinates (features) once, we measure the change in weight for each coordinate. If no coordinate changes by more than a specified threshold, we stop.

For each iteration:
1. As you loop over features in order and perform coordinate descent, measure how much each coordinate changes.
2. After the loop, if the maximum change across all coordinates is falls below the tolerance, stop. Otherwise, go back to step 1.

Return weights

**IMPORTANT: when computing a new weight for coordinate i, we have to make sure to incorporate the new weights for coordinates 0, 1, ..., i-1. One good way is to update your weights variable in-place. See following pseudocode for illustration.**
```
for i in range(len(weights)):
    old_weights_i = weights[i] # remember old value of weight[i], as it will be overwritten
    # the following line uses new values for weight[0], weight[1], ..., weight[i-1]
    #     and old values for weight[i], ..., weight[d-1]
    weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)

    # use old_weights_i to compute change in coordinate
    ...
```


```python
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = initial_weights
    max_weights_change = tolerance
    while (max_weights_change >= tolerance):
        max_weights_change = 0
        for i in range(len(weights)):
            old_weights_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            weights_change = abs(old_weights_i - weights[i])
            if weights_change > max_weights_change:
                max_weights_change = weights_change
    return weights    
```

Now we will use the following parameters, learn the weights on our dataset.


```python
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0
```

First create a normalised version of the feature matrix, `normalized_simple_feature_matrix`


```python
(simple_feature_matrix, output) = get_numpy_data(data, simple_features, my_output)
(normalised_simple_feature_matrix, simple_norms) = normalise_features(simple_feature_matrix) # normalise features
```

Then, run your implementation of LASSO coordinate descent:


```python
weights = lasso_cyclical_coordinate_descent(normalised_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
```

### evaluating LASSO fit with more features

Let's split our dataset into training and test sets.


```python
idx = np.random.rand(len(data)) < 0.8
train = data[idx]; test = data[~idx]
```

Let us consider the following set of features.


```python
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated']
```

First, let's create a normalised feature matrix from the TRAINING data with these features.


```python
(all_feature_matrix, output) = get_numpy_data(train, all_features, my_output)
(normalized_all_feature_matrix, all_norms) = normalise_features(all_feature_matrix)
```

First, let's learn the weights with `l1_penalty=1e7`, on the training data by initialise weights to all zeros, and set the `tolerance=1`.


```python
initial_weights = np.zeros(14)
l1_penalty = 1e7
tolerance = 1.0
weights1e7 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output, initial_weights, l1_penalty, tolerance)
```

Let's see what features had non-zero weight in this case.


```python
weights = zip(['constant'] + all_features, weights1e7)
pd.DataFrame(data=weights, columns=["features","weights"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>constant</td>
      <td>24456466.161094</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bedrooms</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bathrooms</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sqft_living</td>
      <td>48423609.944920</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sqft_lot</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>floors</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>waterfront</td>
      <td>2604909.782880</td>
    </tr>
    <tr>
      <th>7</th>
      <td>view</td>
      <td>6987106.055288</td>
    </tr>
    <tr>
      <th>8</th>
      <td>condition</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>grade</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sqft_above</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sqft_basement</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>yr_built</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>yr_renovated</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Next, let's learn the weights with `l1_penalty=1e8`, on the training data and we will initialise weights to all zeros, and set the `tolerance=1`.


```python
initial_weights = np.zeros(14)
l1_penalty = 1e8
tolerance = 1.0
weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output, initial_weights, l1_penalty, tolerance)
```

Let's investigate what features had non-zero weight in this case.


```python
weights = zip(['constant'] + all_features, weights1e8)
pd.DataFrame(data=weights, columns=["features","weights"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>constant</td>
      <td>71014164.809675</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bedrooms</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bathrooms</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sqft_living</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sqft_lot</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>floors</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>waterfront</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>view</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>condition</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>grade</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sqft_above</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sqft_basement</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>yr_built</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>yr_renovated</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Finally, let's learn the weights with `l1_penalty=1e4`, on the training data. Again we will initialise weights to all zeros, and set the `tolerance=5e5`.  


```python
initial_weights = np.zeros(14)
l1_penalty = 5e5
tolerance = 1.0
weights5e5 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output, initial_weights, l1_penalty, tolerance)
```

And let's see what features had non-zero weight in this case.


```python
weights = zip(['constant'] + all_features, weights5e5)
pd.DataFrame(data=weights, columns=["features","weights"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>constant</td>
      <td>-75627565.515223</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bedrooms</td>
      <td>-8954064.094477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bathrooms</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sqft_living</td>
      <td>55240544.753401</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sqft_lot</td>
      <td>-1252952.799932</td>
    </tr>
    <tr>
      <th>5</th>
      <td>floors</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>waterfront</td>
      <td>5897782.762164</td>
    </tr>
    <tr>
      <th>7</th>
      <td>view</td>
      <td>6157547.970720</td>
    </tr>
    <tr>
      <th>8</th>
      <td>condition</td>
      <td>17003929.039255</td>
    </tr>
    <tr>
      <th>9</th>
      <td>grade</td>
      <td>85659196.432350</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sqft_above</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sqft_basement</td>
      <td>1430748.489541</td>
    </tr>
    <tr>
      <th>12</th>
      <td>yr_built</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>yr_renovated</td>
      <td>3423291.092671</td>
    </tr>
  </tbody>
</table>
</div>



### rescaling learned weights

We normalised our feature matrix before learning the weights.  To use these weights on the test set, we have to normalise the test data in the same way.

Alternatively, we can rescale the learned weights to include the normalisation, so we never have to worry about normalising the test data:

In this case, we need to scale the resulting weights so that we can make predictions with *original* features:
 1. Store the norms of the original features to a vector called `norms`:
```
features, norms = normalise_features(features)
```
 2. Run Lasso on the normalised features and obtain a `weights` vector
 3. Compute the weights for the original features by performing element-wise division, i.e.
```
weights_normalised = weights / norms
```
Now, we can apply `weights_normalised` to the test data, without normalising it!

Create a normalised version of each of the weights learned above. (`weights1e7`, `weights1e8`).


```python
normalised_weights1e7 = weights1e7 / all_norms
normalised_weights1e8 = weights1e8 / all_norms
```

### evaluating each of the learned models on the test data

Let's evaluate the three models on the test data:


```python
(test_feature_matrix, test_output) = get_numpy_data(test, all_features, 'price')
```

Compute the RSS of each of the three normalized weights on the (unnormalised) `test_feature_matrix`:


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
RSS = get_residual_sum_of_squares(test_feature_matrix, test_output, normalised_weights1e7)
print("RSS on TEST data with normalised_weights1e7: $%.6f" % (RSS))
```

    RSS on TEST data with normalised_weights1e7: $291954513074604.375000



```python
RSS = get_residual_sum_of_squares(test_feature_matrix, test_output, normalised_weights1e8)
print("RSS on TEST data with normalised_weights1e8: $%.6f" % (RSS))
```

    RSS on TEST data with normalised_weights1e8: $577660448870259.125000


We can see that RSS of 1e8 is higher tan 1e6

*last edited 10/11/2016*

<a href="#top">Go to top</a>
