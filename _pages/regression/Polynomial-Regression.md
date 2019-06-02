---
permalink: /regression/Polynomial-Regression/
header:
  image: "/images/digital-transition2.jpg"
---
<h2 id="top"></h2>

### Assessing Fit (Polynomial Regression)

In this notebook we will compare different regression models in order to assess which model fits best. We will be using polynomial regression as a mean to examine this topic.

**outline for this notebook**
* In this notebook, we will write a function to take a dataframe and a degree. Then, return a dataframe where each column is the dataframe to a polynomial value up to the total degree e.g. degree = 3 then column 1 is the dataframe, column 2 is the dataframe squared and column 3 is the dataframe cubed
* we will use matplotlib to visualise polynomial regressions
* we will use matplotlib to visualise the same polynomial degree on different subsets of the data
* we will use a validation set to select a polynomial degree
* we will assess the final fit using test data

For this notebook, we will continue to use the House data.

### import library


```python
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
from sklearn import linear_model
import matplotlib.pyplot as plt
```

### read data in

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



In this notebook, we will primarily experiment on "price" and "sqft_living". Therefore, we will sort our data first for latter use.


```python
data = df.sort(['sqft_living', 'price'], ascending=[1, 0])
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
      <th>19452</th>
      <td>3980300371</td>
      <td>20140926T000000</td>
      <td>142000</td>
      <td>0</td>
      <td>0.00</td>
      <td>290</td>
      <td>20875</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>290</td>
      <td>0</td>
      <td>1963</td>
      <td>0</td>
      <td>98024</td>
      <td>47.5308</td>
      <td>-121.888</td>
      <td>1620</td>
      <td>22850</td>
    </tr>
    <tr>
      <th>15381</th>
      <td>2856101479</td>
      <td>20140701T000000</td>
      <td>276000</td>
      <td>1</td>
      <td>0.75</td>
      <td>370</td>
      <td>1801</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>370</td>
      <td>0</td>
      <td>1923</td>
      <td>0</td>
      <td>98117</td>
      <td>47.6778</td>
      <td>-122.389</td>
      <td>1340</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>860</th>
      <td>1723049033</td>
      <td>20140620T000000</td>
      <td>245000</td>
      <td>1</td>
      <td>0.75</td>
      <td>380</td>
      <td>15000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>380</td>
      <td>0</td>
      <td>1963</td>
      <td>0</td>
      <td>98168</td>
      <td>47.4810</td>
      <td>-122.323</td>
      <td>1170</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>18379</th>
      <td>1222029077</td>
      <td>20141029T000000</td>
      <td>265000</td>
      <td>0</td>
      <td>0.75</td>
      <td>384</td>
      <td>213444</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>384</td>
      <td>0</td>
      <td>2003</td>
      <td>0</td>
      <td>98070</td>
      <td>47.4177</td>
      <td>-122.491</td>
      <td>1920</td>
      <td>224341</td>
    </tr>
    <tr>
      <th>21332</th>
      <td>9266700190</td>
      <td>20150511T000000</td>
      <td>245000</td>
      <td>1</td>
      <td>1.00</td>
      <td>390</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>390</td>
      <td>0</td>
      <td>1920</td>
      <td>0</td>
      <td>98103</td>
      <td>47.6938</td>
      <td>-122.347</td>
      <td>1340</td>
      <td>5100</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Next we are going to write a polynomial function that takes a dataframe and a maximal degree. Then, returns a dataframe with columns containing the dataframe to all the powers up to the maximal degree.

The easiest way to apply a power to a dataframe is to use the .apply() and lambda x: functions.
For example to take the example array and compute the third power we can do as follows:


```python
tmp = pd.DataFrame([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print tmp
print tmp_cubed
```

       0
    0  1
    1  2
    2  3
        0
    0   1
    1   8
    2  27


Since the "tmp_cubed" is alrady a dataframe, what we need to do now is to change the column name to be more intuative. In this case, we will name coulumn in accordance with the power it takes.


```python
tmp_cubed.columns = ["power_1"]
tmp_cubed
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>power_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>



### create polynomial dataframe function

We will use what we have tired above to implement dataframe consisting of the powers of a dataframe up to a specific degree:


```python
def polynomial_dataframe(feature, degree):
    # assume that degree >= 1
    # and set polynomial_df['power_1'] equal to the passed feature
    # use deep copy here. otherwise, it will do shallow copy.
    polynomial_df = feature.copy(deep=True)
    polynomial_df.columns = ["power_1"]
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign polynomial_df[name] to the appropriate power of feature
            polynomial_df[name]=feature.apply(lambda x: x**power)
    return polynomial_df
```

To test your function consider the smaller tmp variable and what you would expect the outcome of the following call:


```python
tmp
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
polynomial_dataframe(tmp, 5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>power_1</th>
      <th>power_2</th>
      <th>power_3</th>
      <th>power_4</th>
      <th>power_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>16</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
      <td>27</td>
      <td>81</td>
      <td>243</td>
    </tr>
  </tbody>
</table>
</div>



### visualizing polynomial regression degree 1

Let's start with a degree 1 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.


```python
poly1_data = polynomial_dataframe(pd.DataFrame(data["sqft_living"]), 1)
poly1_data["price"] = data["price"] #add price to the data since it's the target
poly1_data.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>power_1</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8092</th>
      <td>9640</td>
      <td>4668000</td>
    </tr>
    <tr>
      <th>9254</th>
      <td>9890</td>
      <td>6885000</td>
    </tr>
    <tr>
      <th>3914</th>
      <td>10040</td>
      <td>7062500</td>
    </tr>
    <tr>
      <th>7252</th>
      <td>12050</td>
      <td>7700000</td>
    </tr>
    <tr>
      <th>12777</th>
      <td>13540</td>
      <td>2280000</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's make the model from sklearn library


```python
# Create linear regression object
regfit = linear_model.LinearRegression(fit_intercept=True)
# Train the model using the training sets
regfit.fit(poly1_data.iloc[:,0].to_frame(), poly1_data.iloc[:,1].to_frame())
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



Now let's look at the coefficient of our fit


```python
"intercept: ", regfit.intercept_
```




    ('intercept: ', array([-43580.74309448]))




```python
"coefficients: ", regfit.coef_
```




    ('coefficients: ', array([[ 280.6235679]]))



Now we will visualise our data against the fitted line from the model we built.


```python
#predict "power_1"
poly1_data["predicted"] = regfit.predict(poly1_data.iloc[:,0].to_frame())
```


```python
plt.plot(poly1_data['power_1'],poly1_data['price'],'b.',
        poly1_data['power_1'], poly1_data["predicted"],'c-')
```




    [<matplotlib.lines.Line2D at 0x177bb080>,
     <matplotlib.lines.Line2D at 0x177bb278>]




![png](/images/Polynomial-Regression/output_29_1.png)


Let's unpack that plt.plot() command. The first pair of dataframe we passed are the 1st power of sqft and the actual price we then ask it to print these as dots '.'. The next pair we pass is the 1st power of sqft and the predicted values from the linear model. We ask these to be plotted as a line '-'.

We can see, not surprisingly, that the predicted values all fall on a line, specifically the one with slope 280 and intercept -43579. What if we wanted to plot a second degree polynomial?

### visualizing polynomial regression degree 2


```python
poly2_data = polynomial_dataframe(pd.DataFrame(data["sqft_living"]), 2)
poly2_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>power_1</th>
      <th>power_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19452</th>
      <td>290</td>
      <td>84100</td>
    </tr>
    <tr>
      <th>15381</th>
      <td>370</td>
      <td>136900</td>
    </tr>
    <tr>
      <th>860</th>
      <td>380</td>
      <td>144400</td>
    </tr>
    <tr>
      <th>18379</th>
      <td>384</td>
      <td>147456</td>
    </tr>
    <tr>
      <th>21332</th>
      <td>390</td>
      <td>152100</td>
    </tr>
  </tbody>
</table>
</div>




```python
features2 = list(poly2_data.columns.values) # get the name of the features
poly2_data["price"] = data["price"] # add price to the data since it's the target
# Create linear regression object
regfit2 = linear_model.LinearRegression(fit_intercept=True)
#train model
regfit2.fit(poly2_data[features2], poly2_data.iloc[:,(len(poly2_data.columns)-1)].to_frame())
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



Now let's inspect the coefficients


```python
"intercept: ", regfit2.intercept_
```




    ('intercept: ', array([ 199222.27930548]))




```python
"coefficients: ", regfit2.coef_
```




    ('coefficients: ', array([[  6.79940947e+01,   3.85812609e-02]]))



Now we will visualise our data against the fitted line from the model we built.


```python
#let's make the prediction first
poly2_data["predicted"] = regfit2.predict(poly2_data[features2])
```


```python
plt.plot(poly2_data['power_1'],poly2_data['price'],'b.',
        poly2_data['power_1'], poly2_data["predicted"],'c-')
```




    [<matplotlib.lines.Line2D at 0x1aa4a278>,
     <matplotlib.lines.Line2D at 0x1aa4a400>]




![png](/images/Polynomial-Regression/output_39_1.png)


The resulting model looks like half a parabola. Let's try and see what the cubic looks like:

### visualizing polynomial regression degree 3


```python
poly3_data = polynomial_dataframe(pd.DataFrame(data["sqft_living"]), 3)
poly3_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>power_1</th>
      <th>power_2</th>
      <th>power_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19452</th>
      <td>290</td>
      <td>84100</td>
      <td>24389000</td>
    </tr>
    <tr>
      <th>15381</th>
      <td>370</td>
      <td>136900</td>
      <td>50653000</td>
    </tr>
    <tr>
      <th>860</th>
      <td>380</td>
      <td>144400</td>
      <td>54872000</td>
    </tr>
    <tr>
      <th>18379</th>
      <td>384</td>
      <td>147456</td>
      <td>56623104</td>
    </tr>
    <tr>
      <th>21332</th>
      <td>390</td>
      <td>152100</td>
      <td>59319000</td>
    </tr>
  </tbody>
</table>
</div>



Now, we shall train the model.


```python
features3 = list(poly3_data.columns.values) # get the name of the features
print features3
poly3_data['price'] = data['price'] # add price to the data since it's the target
# Create linear regression object
regfit3 = linear_model.LinearRegression(fit_intercept=True)
#train model
regfit3.fit(poly3_data[features3], poly3_data.iloc[:,(len(poly3_data.columns)-1)].to_frame())
```

    ['power_1', 'power_2', 'power_3']





    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



Now we shall make the prediction and plot.


```python
#let's make the prediction first
poly3_data["predicted"] = regfit3.predict(poly3_data[features3])
plt.plot(poly3_data['power_1'],poly3_data['price'],'b.',
        poly3_data['power_1'], poly3_data["predicted"],'c-')
```




    [<matplotlib.lines.Line2D at 0x1abc3978>,
     <matplotlib.lines.Line2D at 0x1abc3b00>]




![png](/images/Polynomial-Regression/output_46_1.png)


### visualizing polynomial regression degree 15

Now let's try a 15th degree polynomial:


```python
poly15_data = polynomial_dataframe(pd.DataFrame(data["sqft_living"]), 15)
poly15_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>power_1</th>
      <th>power_2</th>
      <th>power_3</th>
      <th>power_4</th>
      <th>power_5</th>
      <th>power_6</th>
      <th>power_7</th>
      <th>power_8</th>
      <th>power_9</th>
      <th>power_10</th>
      <th>power_11</th>
      <th>power_12</th>
      <th>power_13</th>
      <th>power_14</th>
      <th>power_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19452</th>
      <td>290</td>
      <td>84100</td>
      <td>24389000</td>
      <td>7072810000</td>
      <td>2051114900000</td>
      <td>594823321000000</td>
      <td>172498763090000000</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
    </tr>
    <tr>
      <th>15381</th>
      <td>370</td>
      <td>136900</td>
      <td>50653000</td>
      <td>18741610000</td>
      <td>6934395700000</td>
      <td>2565726409000000</td>
      <td>949318771330000000</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
    </tr>
    <tr>
      <th>860</th>
      <td>380</td>
      <td>144400</td>
      <td>54872000</td>
      <td>20851360000</td>
      <td>7923516800000</td>
      <td>3010936384000000</td>
      <td>1144155825920000000</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
    </tr>
    <tr>
      <th>18379</th>
      <td>384</td>
      <td>147456</td>
      <td>56623104</td>
      <td>21743271936</td>
      <td>8349416423424</td>
      <td>3206175906594816</td>
      <td>1231171548132409344</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
    </tr>
    <tr>
      <th>21332</th>
      <td>390</td>
      <td>152100</td>
      <td>59319000</td>
      <td>23134410000</td>
      <td>9022419900000</td>
      <td>3518743761000000</td>
      <td>1372310066790000128</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
      <td>-9223372036854775808</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's train the model.


```python
features15 = list(poly15_data.columns.values) # get the name of the features
print features15
poly15_data['price'] = data['price'] # add price to the data since it's the target
# Create linear regression object
regfit15 = linear_model.LinearRegression(fit_intercept=True)
#train model
regfit15.fit(poly15_data[features15], poly15_data.iloc[:,(len(poly15_data.columns)-1)].to_frame())
```

    ['power_1', 'power_2', 'power_3', 'power_4', 'power_5', 'power_6', 'power_7', 'power_8', 'power_9', 'power_10', 'power_11', 'power_12', 'power_13', 'power_14', 'power_15']





    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



lét's look at the plot.


```python
#let's make the prediction first, then plot prediction
poly15_data["predicted"] = regfit15.predict(poly15_data[features15])
plt.plot(poly15_data['power_1'],poly15_data['price'],'b.',
        poly15_data['power_1'], poly15_data["predicted"],'c-')
```




    [<matplotlib.lines.Line2D at 0x1b38f470>,
     <matplotlib.lines.Line2D at 0x1b38f5f8>]




![png](/images/Polynomial-Regression/output_53_1.png)


### selecting a polynomial degree

Whenever we have a "magic" parameter like the degree of the polynomial, there is one well-known way to select these parameters: validation set. We can use validation set to help us find what degree is proper for our data set and also help prevent "overfitting" issue. Therefore, in this section, we will use validation set to help us determine the suitable degree and the indicator that use to measure this is Mean Square Error (MSE).

We split the sales dataset 3-way into training set, test set, and validation set as follows:

* Split our sales data into 2 sets: `training_and_validation` and `testing`.
* Further split our training data into two sets: `training` and `validation`.


```python
idx = np.random.rand(len(data)) < 0.8
train = data[idx]; test = data[~idx]
idx = np.random.rand(len(train)) < 0.8
valid = train[~idx]; train = train[idx]
```

Next we will write a loop that fit model from degree 1 to 15 and compute the "MSE" and store it to dataframe.


```python
#define an empty dataframe to keep track of MSE
mse_df = pd.DataFrame({"degree":[],"mse":[]})

for degree in range(1, 16):
    poly_data = polynomial_dataframe(pd.DataFrame(train["sqft_living"]), degree)
    features = list(poly_data.columns.values) # get the name of the features
    poly_data["price"] = train["price"] # add price to the data since it's the target
    # Create linear regression object
    regfit = linear_model.LinearRegression(fit_intercept=True)
    #train model
    regfit.fit(poly_data[features], poly_data.iloc[:,(len(poly_data.columns)-1)].to_frame())
    #validate the model
    poly_valid = polynomial_dataframe(pd.DataFrame(valid["sqft_living"]), degree)
    poly_valid["price"] = valid["price"] # add price to the data since it's the target
    poly_valid["predicted"] = regfit.predict(poly_valid[features])
    mse = np.mean((poly_valid["predicted"]- poly_valid["price"])**2)
    mse_df = mse_df.append(pd.DataFrame([[degree, mse]], columns=["degree","mse"]))
# inspect mse
mse_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>degree</th>
      <th>mse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6.997170e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>6.381786e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>6.349781e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>6.354785e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>6.345982e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>6.345451e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>6.345192e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>6.345192e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>6.345192e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>6.345192e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>6.345192e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>6.345192e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>13</td>
      <td>6.345193e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>14</td>
      <td>6.345193e+10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>6.345193e+10</td>
    </tr>
  </tbody>
</table>
</div>



Now let's visualse our MSE


```python
plt.plot(mse_df['degree'],mse_df['mse'],'b.-')
```




    [<matplotlib.lines.Line2D at 0x1ab2c748>]




![png](/images/Polynomial-Regression/output_60_1.png)


From the visualisation above, we can see that after degree 5 MSE does not significantly decrease. Therefore we would pick degree 5 as our maximum number for polynormial.

Now that we have chosen the degree of your polynomial using validation data, let's compute the RSS of this model on TEST data.


```python
degree = 5
poly_data = polynomial_dataframe(pd.DataFrame(train["sqft_living"]), degree)
features = list(poly_data.columns.values) # get the name of the features
poly_data["price"] = train["price"] # add price to the data since it's the target
# Create linear regression object
regfit = linear_model.LinearRegression(fit_intercept=True)
#train model
regfit.fit(poly_data[features], poly_data.iloc[:,(len(poly_data.columns)-1)].to_frame())
#test the model
poly_test = polynomial_dataframe(pd.DataFrame(test["sqft_living"]), degree)
poly_test["price"] = test["price"] # add price to the data since it's the target
poly_test["predicted"] = regfit.predict(poly_test[features])
mse = np.mean((poly_test["predicted"]- poly_test["price"])**2)
mse
```




    61167415314.65606



*last edited: 27/10/2016*

<a href="#top">Go to top</a>
