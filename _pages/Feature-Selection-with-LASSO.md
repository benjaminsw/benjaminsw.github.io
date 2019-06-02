---
permalink: /Feature-Selection-with-LASSO/
header:
  image: "/images/digital-transition2.jpg"
---

### Feature Selection and LASSO

In this notebook, we will use LASSO to select features, building on a pre-implemented solver for LASSO.

** outline for this notebook**

* We will run LASSO with different L1 penalties.
* We will choose best L1 penalty using a validation set.
* We will choose best L1 penalty using a validation set, with additional constraint on the size of subset.

In the second notebook, we will implement our own LASSO solver, using coordinate descent.

### import library

In this section, we will import library for later use.


```python
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from math import log, sqrt
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
from sklearn import linear_model
```

### Read data in

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



### create new features

In this notebook, we consider to transform some input features.


```python
data['sqft_living_sqrt'] = data['sqft_living'].apply(sqrt)
data['sqft_lot_sqrt'] = data['sqft_lot'].apply(sqrt)
data['bedrooms_square'] = data['bedrooms']*data['bedrooms']

# In the dataset, 'floors' was defined with type string,
# so we'll convert them to float, before creating a new feature.
data['floors'] = data['floors'].astype(float)
data['floors_square'] = data['floors']*data['floors']
```

* Squaring bedrooms will increase the separation between not many bedrooms (e.g. 1) and lots of bedrooms (e.g. 4) since 1^2 = 1 but 4^2 = 16. Consequently this variable will mostly affect houses with many bedrooms.
* On the other hand, taking square root of sqft_living will decrease the separation between big house and small house. The owner may not be exactly twice as happy for getting a house that is twice as big.

### learn regression weights with L1 penalty

Let'ss fit a model with all the features available in addition to the features we just created above.


```python
all_features = ['bedrooms', 'bedrooms_square','bathrooms', 'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt', 'floors', 'floors_square', 'waterfront', 'view',
            'condition', 'grade','sqft_above','sqft_basement','yr_built', 'yr_renovated']
```

Applying L1 penalty requires `linear_model.Lasso()` from linear_model. In this section we will use `l1 penalty = 1e10` to the linear regression.


```python
regfit = linear_model.Lasso(alpha=1e5)
regfit.fit(data[all_features], data["price"])
```




    Lasso(alpha=100000.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)



Find what features had non-zero weight.


```python
pd.DataFrame(np.append(regfit.intercept_,regfit.coef_), columns=["coefficient"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4622788.430771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1590.124357</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>291.126810</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.067444</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-950.345152</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>38.764596</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-2364.943534</td>
    </tr>
    <tr>
      <th>17</th>
      <td>39.865638</td>
    </tr>
  </tbody>
</table>
</div>



Note that a majority of the weights have been set to zero. So by setting an L1 penalty that's large enough, we are performing a subset selection.

### selecting an L1 penalty

To find a good L1 penalty, we will explore multiple values using a validation set. Let us do three way split into train, validation, and test sets:
* Split our data into 2 sets: training and test
* Further split our training data into two sets: train, validation


```python
idx = np.random.rand(len(data))<0.8
train = data[idx]; test= data[~idx]
idx = np.random.rand(len(train))<0.8
valid = train[~idx]; train= train[idx]
```

Next, we will write a loop that does the following:
* For `l1_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, type `np.logspace(1, 7, num=13)`.)
    * Fit a regression model with a given `l1_penalty` on TRAIN data. Specify `l1_penalty=l1_penalty` and `l2_penalty=0.` in the parameter list.
    * Compute the RSS on VALIDATION data (here you will want to use `.predict()`) for that `l1_penalty`
* Report which `l1_penalty` produced the lowest RSS on validation data.


```python
l1_penality = np.logspace(1, 7, num=13)
print l1_penality
```

    [  1.00000000e+01   3.16227766e+01   1.00000000e+02   3.16227766e+02
       1.00000000e+03   3.16227766e+03   1.00000000e+04   3.16227766e+04
       1.00000000e+05   3.16227766e+05   1.00000000e+06   3.16227766e+06
       1.00000000e+07]



```python
rss_validation =[]
rss_min = float('inf')
for penalty in np.logspace(1, 7, num=13):
    lassofit = linear_model.Lasso(alpha=1e5)
    lassofit.fit(train[all_features], train["price"])
    predicted = lassofit.predict(valid[all_features])
    residules = predicted - valid['price']
    rss = (residules * residules).sum()
    rss_validation.append(rss)
    if rss < rss_min:
        # re-assign new min
        rss_max = rss
        # kepp the best model found so far
        model_with_best_rss = lassofit
#print rss_validation
print 'best rss for validation set', rss_max
```

    best rss for validation set 2.04059895908e+14


Now we will compute RSS from our test set.


```python
residule_test = model_with_best_rss.predict(test[all_features]) - test['price']
print 'RSS from our best model with test data',(residule_test * residule_test).sum()
```

    RSS from our best model with test data 2.82389002842e+14


Now, we will take a look at the coefficients of our model.


```python
pd.DataFrame(np.append(model_with_best_rss.intercept_,model_with_best_rss.coef_), columns=["coefficient"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4579313.014629</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1174.696168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>289.501457</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.048240</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-969.058142</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>43.450123</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-2345.830952</td>
    </tr>
    <tr>
      <th>17</th>
      <td>38.317930</td>
    </tr>
  </tbody>
</table>
</div>



### limit the number of nonzero weights

What if we absolutely wanted to limit ourselves to, say, 7 features? This may be important if we want to derive "a rule of thumb" --- an interpretable model that has only a few features in them.

In this section, you are going to implement a simple, two phase procedure to achive this goal:
1. Explore a large range of `l1_penalty` values to find a narrow region of `l1_penalty` values where models are likely to have the desired number of non-zero weights.
2. Further explore the narrow region you found to find a good value for `l1_penalty` that achieves the desired sparsity.  Here, we will again use a validation set to choose the best value for `l1_penalty`.


```python
max_nonzeros = 7
```

## exploring the larger range of values to find a narrow range with the desired sparsity

Let's define a wide range of possible `l1_penalty_values`:


```python
l1_penalty_values = np.logspace(2, 10, num=20)
l1_penalty_values
```




    array([  1.00000000e+02,   2.63665090e+02,   6.95192796e+02,
             1.83298071e+03,   4.83293024e+03,   1.27427499e+04,
             3.35981829e+04,   8.85866790e+04,   2.33572147e+05,
             6.15848211e+05,   1.62377674e+06,   4.28133240e+06,
             1.12883789e+07,   2.97635144e+07,   7.84759970e+07,
             2.06913808e+08,   5.45559478e+08,   1.43844989e+09,
             3.79269019e+09,   1.00000000e+10])



Now, we will implement a loop that search through this space of possible `l1_penalty` values:

* For `l1_penalty` in `np.logspace(2, 10, num=20)`:
    * Fit a regression model with a given `l1_penalty` on TRAIN data.
    * Extract the weights of the model and count the number of nonzeros. Save the number of nonzeros to a list.


```python
non_zero_l1 = []
for l1 in l1_penalty_values:
    lassofit = linear_model.Lasso(alpha=l1)
    lassofit.fit(train[all_features], train["price"])
    non_zero_l1.append(np.count_nonzero(np.append(lassofit.intercept_, lassofit.coef_)))

print non_zero_l1
```

    [18, 18, 17, 17, 16, 13, 12, 9, 7, 7, 5, 4, 4, 3, 3, 3, 2, 2, 1, 1]


Out of this large range, we want to find the two ends of our desired narrow range of `l1_penalty`.  At one end, we will have `l1_penalty` values that have too few non-zeros, and at the other end, we will have an `l1_penalty` that has too many non-zeros.  

More formally, we wnt to find:
* The largest `l1_penalty` that has more non-zeros than `max_nonzero` and we will store this value in the variable `l1_penalty_min`
* The smallest `l1_penalty` that has fewer non-zeros than `max_nonzero`and we will store this value in the variable `l1_penalty_max`


```python
i = 0
while (non_zero_l1[i] > max_nonzeros):
    i += 1
l1_penalty_min = l1_penalty_values[i - 1]
print 'largest l1 penalty %s with non-zero params more than max-non-zero index %s' % (l1_penalty_min, i-1)
l1_penalty_max = l1_penalty_values[i]
print 'smallest l1 penalty %s with non-zero params less than max-non-zero index %s' % (l1_penalty_max, i)
```

    largest l1 penalty 88586.679041 with non-zero params more than max-non-zero index 7
    smallest l1 penalty 233572.146909 with non-zero params less than max-non-zero index 8


### exploring the narrow range of values to find the solution with the right number of non-zeros that has lowest RSS on the validation set

We will now explore the narrow region of `l1_penalty` values we found:


```python
l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)
print l1_penalty_values
```

    [  88586.67904101   96217.49313932  103848.30723764  111479.12133596
      119109.93543427  126740.74953259  134371.5636309   142002.37772922
      149633.19182754  157264.00592585  164894.82002417  172525.63412248
      180156.4482208   187787.26231912  195418.07641743  203048.89051575
      210679.70461406  218310.51871238  225941.3328107   233572.14690901]


* For `l1_penalty` in `np.linspace(l1_penalty_min,l1_penalty_max,20)`:
    * Fit a regression model with a given `l1_penalty` on TRAIN data. Specify `l1_penalty=l1_penalty` and `l2_penalty=0.` in the parameter list. When you call `linear_regression.create()` make sure you set `validation_set = None`
    * Measure the RSS of the learned model on the VALIDATION set

Now let's find the model that the lowest RSS on the VALIDATION set and has sparsity *equal* to `max_nonzero`.


```python
rss_validation2 =[]
rss_max2 = float('inf')

for l1 in l1_penalty_values:
    lassofit = linear_model.Lasso(alpha=l1)
    lassofit.fit(train[all_features], train["price"])
    if (np.count_nonzero(np.append(lassofit.intercept_, lassofit.coef_)) == max_nonzeros):
        predicted = lassofit.predict(valid[all_features])
        residules = predicted - valid['price']
        rss = (residules * residules).sum()
        rss_validation2.append(rss)
        if rss < rss_max2:
            l1_penalty_with_lowest_rss_max_nonzero = penalty
            rss_max2 = rss
            model_with_best_rss2 = lassofit
print rss_validation2
print 'best rss for validation set in narrow range of l1 penalty', rss_max2
print 'l1_penalty with lowerst rss on validation and max non zero', l1_penalty_with_lowest_rss_max_nonzero
pd.DataFrame(np.append(model_with_best_rss2.intercept_, model_with_best_rss2.coef_), columns=["coefficient"])
```

    [205602567450156.88]
    best rss for validation set in narrow range of l1 penalty 2.0560256745e+14
    l1_penalty with lowerst rss on validation and max non zero 10000000.0





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4092923.302669</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>283.712672</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.813702</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-790.985527</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>38.996378</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-2103.695347</td>
    </tr>
    <tr>
      <th>17</th>
      <td>41.471200</td>
    </tr>
  </tbody>
</table>
</div>



*last edited: 05/11/2016*
