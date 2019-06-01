---
permalink: /Ridge-Regression/
header:
  image: "/images/digital-transition2.jpg"
---

# Ridge Regression

In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization.

**outline for this notebook** <br />
* we will use a pre-built implementation of regression (GraphLab Create) to run polynomial regression
* we will use matplotlib to visualize polynomial regressions
* we will use a pre-built implementation of regression (GraphLab Create) to run polynomial regression, this time with L2 penalty
* we will use matplotlib to visualize polynomial regressions under L2 regularization
* we will choose best L2 penalty using cross-validation.
* we aill assess the final fit using test data.

We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

## import library


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

## read data in

Now, we will read data that we will use in this notebook. This data is house sales in King County, the region where the city of Seattle, WA is located.


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



we will also sort our data by "price" and "sqft_living". This will help us with better visualisation.


```python
data = df.sort(['sqft_living', 'price'], ascending=[1, 1])
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
      <th>4868</th>
      <td>6896300380</td>
      <td>20141002T000000</td>
      <td>228000</td>
      <td>0</td>
      <td>1.00</td>
      <td>390</td>
      <td>5900</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>390</td>
      <td>0</td>
      <td>1953</td>
      <td>0</td>
      <td>98118</td>
      <td>47.5260</td>
      <td>-122.261</td>
      <td>2170</td>
      <td>6000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



## polynomial regression function

Now we will create a polynomial function for later use. This function will create the polynomial of the terget feature up to the given degree.


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

We generate polynomial features up to degree 15 using `polynomial_dataframe()` and fit a model with these features. When fitting the model, we will use an L2 penalty of `1e-5`:


```python
l2_small_penalty = 1e-5
```

Note: When we have so many features and so few data points, the solution can become highly numerically unstable, which can sometimes lead to strange unpredictable results.  Thus, rather than using no regularization, we will introduce a tiny amount of regularization (`l2_penalty=1e-5`) to make the solution numerically stable.  

With the L2 penalty specified above, we will fit the model and print out the learned weights.


```python
def fit15_deg_poly(data, l2_penalty):  
    poly15_data = polynomial_dataframe(pd.DataFrame(data["sqft_living"]), 15)
    features15 = list(poly15_data.columns.values) # get the name of the features
    poly15_data["price"] = data["price"] # add price to the data since it's the target
    # Create linear regression object
    reg15 = Ridge(alpha=l2_penalty, solver='svd')
    #train model
    reg15.fit(poly15_data[features15], poly15_data.iloc[:,(len(poly15_data.columns)-1)].to_frame())
    print("intercept "+str(reg15.intercept_))
    print("coefficient "+str(reg15.coef_))
    #let's make the prediction first, then plot prediction
    poly15_data["predicted"] = reg15.predict(poly15_data[features15])
    plt.plot(poly15_data["power_1"],poly15_data["price"],"b.",
    poly15_data["power_1"], poly15_data["predicted"],"c-")
```


```python
fit15_deg_poly(data, l2_small_penalty)
```

    intercept [ 156830.73690457]
    coefficient [[  1.35096850e+02  -1.09078329e-02   1.16379466e-05  -7.32516116e-10
        7.86140736e-16   7.25121286e-16  -2.67247443e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_16_1.png)


## observe overfitting

The polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the data and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. This is where ridge regression kicks in because it reduces such variance. But first, we will reproduce such cases.

First, split the data into split the data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`.


```python
idx = np.random.rand(len(data))<0.5
semi_split1 = data[idx]; semi_split2 = data[~idx]
idx = np.random.rand(len(semi_split1))<0.5
set_1 = semi_split1[idx]; set_2 = semi_split1[~idx]
idx = np.random.rand(len(semi_split2))<0.5
set_3 = semi_split2[idx]; set_4 = semi_split2[~idx]
```

Next, fit a 15th degree polynomial on `set_1`, `set_2`, `set_3`, and `set_4`, using 'sqft_living' to predict prices. Print the weights and make a plot of the resulting model.


```python
#set 1
fit15_deg_poly(set_1, l2_small_penalty)
```

    intercept [ 214847.27784093]
    coefficient [[  8.79395504e+01  -8.91792713e-03   1.73505356e-05  -1.63867300e-09
       -9.52928073e-15   3.41267009e-16  -1.09453289e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_22_1.png)



```python
#set 2
fit15_deg_poly(set_2, l2_small_penalty)
```

    intercept [ 117345.07723391]
    coefficient [[  2.19657759e+02  -5.45342928e-02   2.00656617e-05  -1.20189180e-09
       -1.04445880e-14   3.91844475e-16  -1.28687054e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_23_1.png)



```python
#set 3
fit15_deg_poly(set_3, l2_small_penalty)
```

    intercept [ 274777.57344825]
    coefficient [[ -1.79365225e+02   1.48652979e-01  -2.01886970e-05   1.32309772e-09
        2.69638258e-14   5.23834725e-16  -1.12635400e-14   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_24_1.png)



```python
#set 4
fit15_deg_poly(set_4, l2_small_penalty)
```

    intercept [ 275790.92011891]
    coefficient [[ -9.87019267e+01   1.04674177e-01  -9.05370886e-06   4.31084780e-10
       -1.97719200e-14   3.26295135e-16  -5.90832396e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_25_1.png)


The four curves should differ from one another a lot, as should the coefficients you learned.

## ridge regression comes to rescue

Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalising "large" weights. Note that weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.

With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set_1`, `set_2`, `set_3`, and `set_4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above.


```python
#define new l2 penaly
l2_penalty = 1e10
```


```python
#set 1
fit15_deg_poly(set_1, l2_penalty)
```

    intercept [ 250818.50246403]
    coefficient [[  8.81891273e-02   3.30201687e-02   9.44020901e-06  -1.14140943e-09
       -8.46120591e-15  -1.61276169e-17  -3.47763588e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_30_1.png)



```python
#set 2
fit15_deg_poly(set_2, l2_penalty)
```

    intercept [ 215546.17905305]
    coefficient [[  4.51670991e-01   2.93502739e-02   8.46473490e-06  -7.19484779e-10
       -1.95592769e-14  -1.39990461e-15  -8.48861065e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_31_1.png)



```python
#set 3
fit15_deg_poly(set_3, l2_penalty)
```

    intercept [ 187052.22395021]
    coefficient [[ -2.49140949e-01   7.12380348e-02  -7.53020689e-06   6.58253134e-10
        3.17578659e-14   1.49678286e-15  -7.20349010e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_32_1.png)



```python
#set 4
fit15_deg_poly(set_4, l2_penalty)
```

    intercept [ 229258.33434574]
    coefficient [[ -1.74749728e-01   6.50438805e-02  -3.20023745e-06   1.65131759e-10
       -1.60408116e-14   9.99168000e-16  -3.15793547e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_33_1.png)



```python
#set 4
fit15_deg_poly(set_4, l2_small_penalty)
```

    intercept [ 275790.92011891]
    coefficient [[ -9.87019267e+01   1.04674177e-01  -9.05370886e-06   4.31084780e-10
       -1.97719200e-14   3.26295135e-16  -5.90832396e-15   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   0.00000000e+00   0.00000000e+00]]



![png](/images/Ridge-Regression/output_34_1.png)


## selecting an L2 penalty via cross-validation

As seen, with randon L2 parameter, we cannot see obvious improvement in our dataset. The L2 penalty is a "magic" parameter we need to select. We could use the validation set approach but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.

We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:

> Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
> Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
> ...<br>
> Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set

After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data.

To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments.


```python
train_valid_shuffled = data.iloc[np.random.permutation(len(data))]
train_valid_shuffled.head()
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
      <th>21183</th>
      <td>5528600005</td>
      <td>20150327T000000</td>
      <td>272167</td>
      <td>2</td>
      <td>2.50</td>
      <td>1620</td>
      <td>3795</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1620</td>
      <td>0</td>
      <td>2014</td>
      <td>0</td>
      <td>98027</td>
      <td>47.5321</td>
      <td>-122.034</td>
      <td>1620</td>
      <td>6000</td>
    </tr>
    <tr>
      <th>8673</th>
      <td>4151800530</td>
      <td>20141028T000000</td>
      <td>1090000</td>
      <td>4</td>
      <td>2.50</td>
      <td>2780</td>
      <td>6837</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>2780</td>
      <td>0</td>
      <td>2004</td>
      <td>0</td>
      <td>98033</td>
      <td>47.6660</td>
      <td>-122.201</td>
      <td>1160</td>
      <td>6837</td>
    </tr>
    <tr>
      <th>9277</th>
      <td>8129700644</td>
      <td>20140703T000000</td>
      <td>715000</td>
      <td>3</td>
      <td>4.00</td>
      <td>2080</td>
      <td>2250</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>8</td>
      <td>2080</td>
      <td>0</td>
      <td>1997</td>
      <td>0</td>
      <td>98103</td>
      <td>47.6598</td>
      <td>-122.355</td>
      <td>2080</td>
      <td>2250</td>
    </tr>
    <tr>
      <th>6677</th>
      <td>2795000080</td>
      <td>20140919T000000</td>
      <td>535100</td>
      <td>3</td>
      <td>2.25</td>
      <td>2070</td>
      <td>7207</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1720</td>
      <td>350</td>
      <td>1973</td>
      <td>0</td>
      <td>98177</td>
      <td>47.7735</td>
      <td>-122.371</td>
      <td>2350</td>
      <td>7980</td>
    </tr>
    <tr>
      <th>1369</th>
      <td>3374300070</td>
      <td>20140623T000000</td>
      <td>334000</td>
      <td>4</td>
      <td>1.50</td>
      <td>1150</td>
      <td>9360</td>
      <td>1.5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>1150</td>
      <td>0</td>
      <td>1970</td>
      <td>0</td>
      <td>98034</td>
      <td>47.7197</td>
      <td>-122.173</td>
      <td>1480</td>
      <td>8155</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.


```python
n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)
```

    0 (0, 2160)
    1 (2161, 4321)
    2 (4322, 6482)
    3 (6483, 8644)
    4 (8645, 10805)
    5 (10806, 12966)
    6 (12967, 15128)
    7 (15129, 17289)
    8 (17290, 19450)
    9 (19451, 21612)


Let's familiarise ourselves with array slicing with dataframe. To extract a continuous slice from an data, use colon in square brackets. For instance, the following cell extracts rows 0 to 9 of `train_valid_shuffled`. Notice that the first index (0) is included in the slice but the last index (10) is omitted.


```python
tmp_df = train_valid_shuffled[0:10] # rows 0 to 9
tmp_df
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
      <th>21183</th>
      <td>5528600005</td>
      <td>20150327T000000</td>
      <td>272167</td>
      <td>2</td>
      <td>2.50</td>
      <td>1620</td>
      <td>3795</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1620</td>
      <td>0</td>
      <td>2014</td>
      <td>0</td>
      <td>98027</td>
      <td>47.5321</td>
      <td>-122.034</td>
      <td>1620</td>
      <td>6000</td>
    </tr>
    <tr>
      <th>8673</th>
      <td>4151800530</td>
      <td>20141028T000000</td>
      <td>1090000</td>
      <td>4</td>
      <td>2.50</td>
      <td>2780</td>
      <td>6837</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>2780</td>
      <td>0</td>
      <td>2004</td>
      <td>0</td>
      <td>98033</td>
      <td>47.6660</td>
      <td>-122.201</td>
      <td>1160</td>
      <td>6837</td>
    </tr>
    <tr>
      <th>9277</th>
      <td>8129700644</td>
      <td>20140703T000000</td>
      <td>715000</td>
      <td>3</td>
      <td>4.00</td>
      <td>2080</td>
      <td>2250</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>8</td>
      <td>2080</td>
      <td>0</td>
      <td>1997</td>
      <td>0</td>
      <td>98103</td>
      <td>47.6598</td>
      <td>-122.355</td>
      <td>2080</td>
      <td>2250</td>
    </tr>
    <tr>
      <th>6677</th>
      <td>2795000080</td>
      <td>20140919T000000</td>
      <td>535100</td>
      <td>3</td>
      <td>2.25</td>
      <td>2070</td>
      <td>7207</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1720</td>
      <td>350</td>
      <td>1973</td>
      <td>0</td>
      <td>98177</td>
      <td>47.7735</td>
      <td>-122.371</td>
      <td>2350</td>
      <td>7980</td>
    </tr>
    <tr>
      <th>1369</th>
      <td>3374300070</td>
      <td>20140623T000000</td>
      <td>334000</td>
      <td>4</td>
      <td>1.50</td>
      <td>1150</td>
      <td>9360</td>
      <td>1.5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>1150</td>
      <td>0</td>
      <td>1970</td>
      <td>0</td>
      <td>98034</td>
      <td>47.7197</td>
      <td>-122.173</td>
      <td>1480</td>
      <td>8155</td>
    </tr>
    <tr>
      <th>13586</th>
      <td>1226059101</td>
      <td>20140701T000000</td>
      <td>502000</td>
      <td>3</td>
      <td>2.25</td>
      <td>1600</td>
      <td>45613</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1600</td>
      <td>0</td>
      <td>1983</td>
      <td>0</td>
      <td>98072</td>
      <td>47.7523</td>
      <td>-122.117</td>
      <td>2320</td>
      <td>43005</td>
    </tr>
    <tr>
      <th>16297</th>
      <td>9346900170</td>
      <td>20140922T000000</td>
      <td>615000</td>
      <td>4</td>
      <td>2.25</td>
      <td>2330</td>
      <td>7020</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1450</td>
      <td>880</td>
      <td>1973</td>
      <td>0</td>
      <td>98006</td>
      <td>47.5620</td>
      <td>-122.139</td>
      <td>2330</td>
      <td>8500</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>2197600451</td>
      <td>20141105T000000</td>
      <td>631000</td>
      <td>5</td>
      <td>2.00</td>
      <td>2270</td>
      <td>2400</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>2270</td>
      <td>0</td>
      <td>1905</td>
      <td>0</td>
      <td>98122</td>
      <td>47.6051</td>
      <td>-122.319</td>
      <td>1320</td>
      <td>2400</td>
    </tr>
    <tr>
      <th>15516</th>
      <td>579000595</td>
      <td>20140906T000000</td>
      <td>724000</td>
      <td>2</td>
      <td>1.00</td>
      <td>1560</td>
      <td>5000</td>
      <td>1.5</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>7</td>
      <td>1560</td>
      <td>0</td>
      <td>1942</td>
      <td>0</td>
      <td>98117</td>
      <td>47.7006</td>
      <td>-122.386</td>
      <td>2620</td>
      <td>5400</td>
    </tr>
    <tr>
      <th>3013</th>
      <td>3423049209</td>
      <td>20150318T000000</td>
      <td>200450</td>
      <td>3</td>
      <td>1.00</td>
      <td>970</td>
      <td>9130</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>970</td>
      <td>0</td>
      <td>1957</td>
      <td>0</td>
      <td>98188</td>
      <td>47.4369</td>
      <td>-122.272</td>
      <td>1000</td>
      <td>8886</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 21 columns</p>
</div>



Now let's extract individual segments with array slicing. we should consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
Just for fun, let's extract the fourth segment (segment 3) and assign it to a variable called `validation4`.

Now we are ready to implement k-fold cross-validation. We will write a function that computes k validation errors by designating each of the k segments as the validation set. It will accept parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function will return the average validation error using k segments as validation sets.

* For each i in [0, 1, ..., k-1]:
  > * Compute starting and ending indices of segment i and call 'start' and 'end'
  > * Form validation set by taking a slice (start:end+1) from the data.
  > * Form training set by appending slice (end+1:n) to the end of slice (0:start).
  > * Train a linear model using training set just formed, with a given l2_penalty
  > * Compute validation error using validation set just formed


```python
def k_fold_cross_validation(k, l2_penalty, data, output, features_list):
    errors = []
    n = len(data)
    for i in xrange(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        # get first fold
        valid = data[start:end]
        first_fold = data[0:start-1]
        remainder_fold = data[end+1:]
        train = first_fold.append(remainder_fold)
        # train model
        # create linear regression object
        reg = Ridge(alpha=l2_penalty, solver='svd')
        # train model
        reg.fit(train[features_list], train.iloc[:,(len(train.columns)-1)].to_frame())
        # prediction
        predicted = reg.predict(valid[features_list])
        sse = (predicted- valid[output])**2
        errors.append(np.mean(sse))
    mse = np.mean(list(errors))
    #print("l2_penalty: %s, \n\t Average MSE: $%.6f" % (l2_penalty, mse))
    return mse
```

Now that we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Now let's write a loop that does the following:
* We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
* For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
    * Run 10-fold cross-validation with `l2_penalty`
* Investigate which L2 penalty produced the lowest average validation error.

Note: since the degree of the polynomial is now fixed to 15, to make things faster, we should generate polynomial features in advance and re-use them throughout the loop.
Note2: make sure to use `train_valid_shuffled` when generating polynomial features!


```python
poly15_data = polynomial_dataframe(pd.DataFrame(train_valid_shuffled["sqft_living"]), 15)
fifteen_features = list(poly15_data.columns.values)# get the name of the features
poly15_data['price'] = train_valid_shuffled['price'] # add price to the data since it's the target

results = []

for l2_penalty in np.logspace(1, 7, num=13):
    average_error = k_fold_cross_validation(10, l2_penalty, poly15_data, ['price'], fifteen_features)
    results.append((l2_penalty, average_error))
mse_df = pd.DataFrame(results, columns=["penalty", "error"])
mse_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>penalty</th>
      <th>error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.000000</td>
      <td>6.630754e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31.622777</td>
      <td>6.630754e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100.000000</td>
      <td>6.630754e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>316.227766</td>
      <td>6.630754e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000.000000</td>
      <td>6.630753e+10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3162.277660</td>
      <td>6.630750e+10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10000.000000</td>
      <td>6.630742e+10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>31622.776602</td>
      <td>6.630716e+10</td>
    </tr>
    <tr>
      <th>8</th>
      <td>100000.000000</td>
      <td>6.630633e+10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>316227.766017</td>
      <td>6.630372e+10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1000000.000000</td>
      <td>6.629560e+10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3162277.660168</td>
      <td>6.627113e+10</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10000000.000000</td>
      <td>6.620413e+10</td>
    </tr>
  </tbody>
</table>
</div>



it will be useful to plot the k-fold cross-validation errors you have obtained to better understand the behavior of the method.  


```python
plt.plot(mse_df['penalty'],mse_df['error'],'b.-')
plt.xscale('log')
plt.xlabel('log(l2_penalty)')
plt.ylabel('average_error')
plt.title('k-fold cross-validation errors')
```




    <matplotlib.text.Text at 0x1c672d68>




![png](/images/Ridge-Regression/output_49_1.png)


*last edit 29/10/2016*
