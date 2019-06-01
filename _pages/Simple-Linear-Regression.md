---
permalink: /Simple-Linear-Regression/
header:
  image: "/images/digital-transition2.jpg"
---

# Simple Linear Regression

In this notebook I use data on house sales in King County to predict house prices using simple (one input) linear regression.
* I will use pandas descripttive to compute summary statistics
* I will compute the Simple Linear Regression weights using the closed form solution
* I will make predictions of the output given the input feature
* I will also try turning the regression around to predict the input given the output
* Finally, I will compare two different models for predicting house prices

# Import library


```python
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
```

# Load house sales data

Dataset is from house sales in King County, the region where the city of Seattle, WA is located.

The following section is to load the entire dataset in and made a dictionary of column names and data type as this was used latter when importing training datset and test dataset


```python
data = pd.read_csv("kc_house_data.csv")
colname_lst = list(data.columns.values)
coltype_lst =  [str, str, float, float, float, float, int, str, int, int, int, int, int, int, int, int, str, float, float, float, float]
col_type_dict = dict(zip(colname_lst, coltype_lst))
```

# Split data into training data and test data

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
      <th>5</th>
      <td>7237550310</td>
      <td>20140512T000000</td>
      <td>1225000</td>
      <td>4</td>
      <td>4.50</td>
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
    <tr>
      <th>6</th>
      <td>1321400060</td>
      <td>20140627T000000</td>
      <td>257500</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2</td>
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
    <tr>
      <th>10</th>
      <td>1736800520</td>
      <td>20150403T000000</td>
      <td>662500</td>
      <td>3</td>
      <td>2.50</td>
      <td>3560</td>
      <td>9796</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1860</td>
      <td>1700</td>
      <td>1965</td>
      <td>0</td>
      <td>98007</td>
      <td>47.6007</td>
      <td>-122.145</td>
      <td>2210</td>
      <td>8925</td>
    </tr>
    <tr>
      <th>18</th>
      <td>16000397</td>
      <td>20141205T000000</td>
      <td>189000</td>
      <td>2</td>
      <td>1.00</td>
      <td>1200</td>
      <td>9850</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1200</td>
      <td>0</td>
      <td>1921</td>
      <td>0</td>
      <td>98002</td>
      <td>47.3089</td>
      <td>-122.210</td>
      <td>1060</td>
      <td>5095</td>
    </tr>
    <tr>
      <th>23</th>
      <td>8091400200</td>
      <td>20140516T000000</td>
      <td>252700</td>
      <td>2</td>
      <td>1.50</td>
      <td>1070</td>
      <td>9643</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1070</td>
      <td>0</td>
      <td>1985</td>
      <td>0</td>
      <td>98030</td>
      <td>47.3533</td>
      <td>-122.166</td>
      <td>1220</td>
      <td>8386</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# Now let's look at descriptive statistics for this data set


```python
train.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
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
      <th>count</th>
      <td>1.726500e+04</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
      <td>17265.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.571240e+09</td>
      <td>539130.778859</td>
      <td>3.373183</td>
      <td>2.114292</td>
      <td>2081.189458</td>
      <td>15239.504663</td>
      <td>1.495743</td>
      <td>0.007356</td>
      <td>0.235042</td>
      <td>3.409962</td>
      <td>7.658384</td>
      <td>1788.401390</td>
      <td>292.788068</td>
      <td>1971.046858</td>
      <td>82.659485</td>
      <td>98077.574109</td>
      <td>47.559089</td>
      <td>-122.213586</td>
      <td>1989.928294</td>
      <td>12918.148856</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.875593e+09</td>
      <td>363000.241270</td>
      <td>0.937721</td>
      <td>0.767328</td>
      <td>916.391531</td>
      <td>41507.204789</td>
      <td>0.541861</td>
      <td>0.085453</td>
      <td>0.767452</td>
      <td>0.650456</td>
      <td>1.172731</td>
      <td>827.495956</td>
      <td>444.716563</td>
      <td>29.354455</td>
      <td>397.708493</td>
      <td>53.492064</td>
      <td>0.138561</td>
      <td>0.140198</td>
      <td>688.513788</td>
      <td>28142.809544</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>75000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>290.000000</td>
      <td>572.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>290.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>460.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.114700e+09</td>
      <td>323000.000000</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5065.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>0.000000</td>
      <td>1952.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.469000</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.904100e+09</td>
      <td>450000.000000</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1920.000000</td>
      <td>7620.000000</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>0.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.571000</td>
      <td>-122.229000</td>
      <td>1850.000000</td>
      <td>7621.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.305300e+09</td>
      <td>641250.000000</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>10766.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>560.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98117.000000</td>
      <td>47.677300</td>
      <td>-122.124000</td>
      <td>2370.000000</td>
      <td>10123.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>7700000.000000</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1651359.000000</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>4820.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Data visualisation

In addition to descriptive statistics, data visualisation also helps to understand the gist of data. Therefore we will spend a bit of time looking at some of the variables, we interested.


```python
sns.distplot(train["price"]);
```


![png](output_15_0.png)



```python
sns.distplot(train["bedrooms"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x9f1f2b0>




![png](output_16_1.png)



```python
sns.distplot(train["bathrooms"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17945710>




![png](output_17_1.png)



```python
sns.distplot(train["sqft_living"])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a5e47f0>




![png](output_18_1.png)


# Build a generic simple linear regression function

This section we will create generic simple linear regression function. This function will take two parameters.
* one parameter is for input variable
* the other one is for output variable

aim: we want to find the line that give the least residual sum of square (RSS).
$$RSS(w_0,w_1) = \sum_{i=1}^{N} (y_i-[w_0+w_1x_i])^{2} $$

take derivation of the above function gives:
$$\frac{\partial RSS(w_0,w_1)}{\partial w_i} = \begin{bmatrix}-2\sum_{i=1}^{N}  (y_i-[w_0+w_1x_i])\\ -2\sum_{i=1}^{N}  (y_i-[w_0+w_1x_i])x_i\end{bmatrix} $$

Then, we set gradient to 0. <br />
Top term will be:
$$ \hat{w_0} = \frac{ \sum_{i=1}^{N} y_i}{N} - \frac{\hat{w_1} \sum_{i=1}^{N} x_i} {N}$$

Bottom term will be:
$$ \hat{w_1} = \frac {\sum y_ix_i - \frac{\sum y_i \sum x_i}{N}}{ \sum x_i^2 - \frac {(\sum x_i)^2}{N} }$$

which these will be implemented below.


```python
def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    input_sum = input_feature.sum()
    output_sum = output.sum()
    # compute the product of the output and the input_feature and its sum
    product_of_input_output = (input_feature*output).sum()
    # compute the squared value of the input_feature and its sum
    sum_of_sqrt_input = (input_feature*input_feature).sum()
    # use the formula for the slope or the top term
    slope = (product_of_input_output-(input_sum*output_sum/output.size))/(sum_of_sqrt_input-input_sum*input_sum/output.size)
    # use the formula for the intercept or the bottom term
    intercept = output_sum/output.size - slope*input_sum/output.size
    return (intercept, slope)
```

We can test that our function works by passing it something where we know the answer. In particular we can generate a feature and then put the output exactly on a line: output = 1 + 1\*input_feature then we know both our slope and intercept should be 1


```python
test_feature = pd.DataFrame(range(5))
test_output = pd.DataFrame(1+1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print "Intercept: " + str(test_intercept[0])
print "Slope: " + str(test_slope[0])
```

    Intercept: 1.0
    Slope: 1.0


Now that we know it works. Let's build a regression model for predicting price based on sqft_living. Rembember that we train on train_data!


```python
sqft_intercept, sqft_slope = simple_linear_regression(train['sqft_living'], train['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)
```

    Intercept: -39378.4243047
    Slope: 277.970465792


# Predicting Values

Now that we have the model parameters: intercept & slope we can make predictions. The following function will return the predicted output given the input_feature, slope and intercept:


```python
def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = intercept + slope*input_feature
    return predicted_values
```

Now that we can calculate a prediction given the slope and intercept let's make a prediction. Now we will predict the estimated price for a house with 2650 squarefeet according to the squarefeet model we estiamted above.


```python
my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)
```

    The estimated price for a house with 2650 squarefeet is $697243.31


# Residual Sum of Squares

Now that we have a model and can make predictions let's evaluate our model using Residual Sum of Squares (RSS). Recall that RSS is the sum of the squares of the residuals and the residuals is just a fancy word for the difference between the predicted output and the true output.

The following function to compute the RSS of a simple linear regression model given the input_feature, output, intercept and slope:


```python
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predicted_values_series = intercept + slope*input_feature
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residual_series = output - predicted_values_series
    # square the residuals and add them up
    residual_square_series = residual_series * residual_series
    RSS = residual_square_series.sum()
    return(RSS)
```

Let's test our get_residual_sum_of_squares function by applying it to the test model where the data lie exactly on a line. Since they lie exactly on a line the residual sum of squares should be zero!


```python
print get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope)
```

    0    0
    dtype: float64


Now use your function to calculate the RSS on training data from the squarefeet model calculated above.


```python
rss_prices_on_sqft = get_residual_sum_of_squares(train['sqft_living'], train['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)
```

    The RSS of predicting Prices based on Square Feet is : 1.15464936338e+15


# Predict the squarefeet given price

What if we want to predict the squarefoot given the price? Since we have an equation y = a + b\*x we can solve the function for x. So that if we have the intercept (a) and the slope (b) and the price (y) we can solve for the estimated squarefeet (x).

The following function will predict the input_feature given the output!


```python
def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output-intercept)/slope
    return estimated_feature
```

Now that we have a function to compute the squarefeet given the price from our simple regression model let's see how big we might expect a house that costs $800,000 to be.


```python
my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)
```

    The estimated squarefeet for a house worth $800000.00 is 3019


# New Model: estimate prices from bedrooms

We have made one model for predicting house prices using squarefeet, but there are many other features in the sales dataframe.
Now we will make a simple linear regression function to estimate the regression parameters from predicting Prices based on number of bedrooms.


```python
# Estimate the slope and intercept for predicting 'price' based on 'bedrooms'
(bedrooms_intercept, bedrooms_slope) =  simple_linear_regression(train['bedrooms'], train['price'])
(bedrooms_intercept, bedrooms_slope)
```




    (135689.62284521962, 119602.51997969164)



# Test our Linear Regression Algorithm

Now we have two models for predicting the price of a house. How do we know which one is better? Calculate the RSS on the TEST data (remember this data wasn't involved in learning the model). Let's compute the RSS from predicting prices using bedrooms and from predicting prices using squarefeet.


```python
# Compute RSS when using squarefeet on TEST data:
rss_prices_on_sqft = get_residual_sum_of_squares(test['sqft_living'], test['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)
```

    The RSS of predicting Prices based on Square Feet is : 3.22792745249e+14



```python
# Compute RSS when using bedrooms on TEST data:
rss_prices_on_sqft = get_residual_sum_of_squares(test['bedrooms'], test['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Bedrooms is : ' + str(rss_prices_on_sqft)
```

    The RSS of predicting Prices based on Bedrooms is : 2.11218065967e+15


*last edit:23/10/16 *
