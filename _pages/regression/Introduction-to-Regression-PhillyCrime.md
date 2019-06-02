---
permalink: /regression/Introduction-to-Regression-PhillyCrime/
header:
  image: "/images/digital-transition2.jpg"
---
<h2 id="top"></h2>

### Introduction to Regression

### import library


```python
%matplotlib inline
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
plt.rcParams["figure.figsize"] = [12,9]
```

    :0: FutureWarning: IPython widgets are experimental and may change in the future.


### Load some house value vs. crime rate data

Dataset is from Philadelphia, PA and includes average house sales price in a number of neighborhoods.  The attributes of each neighborhood we have include the crime rate ('CrimeRate'), miles from Center City ('MilesPhila'), town name ('Name'), and county name ('County').


```python
sales = pd.read_csv('Philadelphia_Crime_Rate_noNA.csv')
```


```python
sales.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HousePrice</th>
      <th>HsPrc ($10,000)</th>
      <th>CrimeRate</th>
      <th>MilesPhila</th>
      <th>PopChg</th>
      <th>Name</th>
      <th>County</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140463</td>
      <td>14.0463</td>
      <td>29.7</td>
      <td>10</td>
      <td>-1.0</td>
      <td>Abington</td>
      <td>Montgome</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113033</td>
      <td>11.3033</td>
      <td>24.1</td>
      <td>18</td>
      <td>4.0</td>
      <td>Ambler</td>
      <td>Montgome</td>
    </tr>
    <tr>
      <th>2</th>
      <td>124186</td>
      <td>12.4186</td>
      <td>19.5</td>
      <td>25</td>
      <td>8.0</td>
      <td>Aston</td>
      <td>Delaware</td>
    </tr>
    <tr>
      <th>3</th>
      <td>110490</td>
      <td>11.0490</td>
      <td>49.4</td>
      <td>25</td>
      <td>2.7</td>
      <td>Bensalem</td>
      <td>Bucks</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79124</td>
      <td>7.9124</td>
      <td>54.1</td>
      <td>19</td>
      <td>3.9</td>
      <td>Bristol B.</td>
      <td>Bucks</td>
    </tr>
    <tr>
      <th>5</th>
      <td>92634</td>
      <td>9.2634</td>
      <td>48.6</td>
      <td>20</td>
      <td>0.6</td>
      <td>Bristol T.</td>
      <td>Bucks</td>
    </tr>
    <tr>
      <th>6</th>
      <td>89246</td>
      <td>8.9246</td>
      <td>30.8</td>
      <td>15</td>
      <td>-2.6</td>
      <td>Brookhaven</td>
      <td>Delaware</td>
    </tr>
    <tr>
      <th>7</th>
      <td>195145</td>
      <td>19.5145</td>
      <td>10.8</td>
      <td>20</td>
      <td>-3.5</td>
      <td>Bryn Athyn</td>
      <td>Montgome</td>
    </tr>
    <tr>
      <th>8</th>
      <td>297342</td>
      <td>29.7342</td>
      <td>20.2</td>
      <td>14</td>
      <td>0.6</td>
      <td>Bryn Mawr</td>
      <td>Montgome</td>
    </tr>
    <tr>
      <th>9</th>
      <td>264298</td>
      <td>26.4298</td>
      <td>20.4</td>
      <td>26</td>
      <td>6.0</td>
      <td>Buckingham</td>
      <td>Bucks</td>
    </tr>
  </tbody>
</table>
</div>



### Exploring the data

The house price in a town is correlated with the crime rate of that town. Low crime towns tend to be associated with higher house prices and vice versa.


```python
sns.lmplot('CrimeRate', 'HousePrice', data=sales, fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0xa3af3c8>




![png](/images/Introduction-to-Regression-PhillyCrime/output_8_1.png)


### Fit the regression model using crime as the feature


```python
sales_X_crimerate  = sales['CrimeRate'].reshape(sales['CrimeRate'].shape[0],1)
sales_y_houseprice = sales['HousePrice'].reshape(sales['HousePrice'].shape[0],1)
```


```python
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the datset
regr.fit(sales_X_crimerate, sales_y_houseprice)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



### Let's see what our fit looks like


```python
sns.lmplot('CrimeRate', 'HousePrice', data=sales)
```




    <seaborn.axisgrid.FacetGrid at 0x187795f8>




![png](/images/Introduction-to-Regression-PhillyCrime/output_13_1.png)


Above: dots are original data, blue line is the fit from the simple regression.

### Remove Center City and redo the analysis

Center City is the one observation with an extremely high crime rate, yet house prices are not very low.  This point does not follow the trend of the rest of the data very well.  A question is how much including Center City is influencing our fit on the other datapoints.  Let's remove this datapoint and see what happens.


```python
sales_noCC = sales[sales['MilesPhila'] != 0.0]
```


```python
sns.lmplot('CrimeRate', 'HousePrice', data=sales_noCC, fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x18c1ce48>




![png](/images/Introduction-to-Regression-PhillyCrime/output_18_1.png)


### Refit our simple regression model on this modified dataset:


```python
sales_noCC_X_crimerate  = sales_noCC['CrimeRate'].reshape(sales_noCC['CrimeRate'].shape[0],1)
sales_noCC_y_houseprice = sales_noCC['HousePrice'].reshape(sales_noCC['HousePrice'].shape[0],1)


# Create linear regression object
regr_noCC = linear_model.LinearRegression()

# Train the model using the training sets
regr_noCC.fit(sales_noCC_X_crimerate, sales_noCC_y_houseprice)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



### Look at the fit:


```python
sns.lmplot('CrimeRate', 'HousePrice', data=sales_noCC)
```




    <seaborn.axisgrid.FacetGrid at 0x18c27358>




![png](/images/Introduction-to-Regression-PhillyCrime/output_22_1.png)


### Compare coefficients for full-data fit versus no-Center-City fit

Visually, the fit seems different, but let's quantify this by examining the estimated coefficients of our original fit and that of the modified dataset with Center City removed.


```python
params_dict = {'interecpt':regr.intercept_[0],'CrimeRate':regr.coef_[0][0]}
pd.DataFrame(params_dict.items(), columns=['name','value'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>interecpt</td>
      <td>176629.408107</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CrimeRate</td>
      <td>-576.908128</td>
    </tr>
  </tbody>
</table>
</div>




```python
params_noCC_dict = {'interecpt':regr_noCC.intercept_[0],'CrimeRate':regr_noCC.coef_[0][0]}
pd.DataFrame(params_noCC_dict.items(), columns=['name','value'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>interecpt</td>
      <td>225233.551839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CrimeRate</td>
      <td>-2288.689430</td>
    </tr>
  </tbody>
</table>
</div>



Above: We see that for the "no Center City" version, per unit increase in crime, the predicted decrease in house prices is 2,287.  In contrast, for the original dataset, the drop is only 576 per unit increase in crime.  This is significantly different!

### High leverage points:
Center City is said to be a "high leverage" point because it is at an extreme x value where there are not other observations.  As a result, recalling the closed-form solution for simple regression, this point has the *potential* to dramatically change the least squares line since the center of x mass is heavily influenced by this one point and the least squares line will try to fit close to that outlying (in x) point.  If a high leverage point follows the trend of the other data, this might not have much effect.  On the other hand, if this point somehow differs, it can be strongly influential in the resulting fit.

### Influential observations:  
An influential observation is one where the removal of the point significantly changes the fit.  As discussed above, high leverage points are good candidates for being influential observations, but need not be.  Other observations that are *not* leverage points can also be influential observations (e.g., strongly outlying in y even if x is a typical value).

### Remove high-value outlier neighborhoods and redo analysis

Based on the discussion above, a question is whether the outlying high-value towns are strongly influencing the fit.  Let's remove them and see what happens.


```python
#remove outliying-value towns
sales_nohighend = sales_noCC[sales_noCC['HousePrice'] < 350000]

sales_noCCnohighend_X_crimerate  = sales_nohighend['CrimeRate'].reshape(sales_nohighend['CrimeRate'].shape[0],1)
sales_noCCnohighend_y_houseprice = sales_nohighend['HousePrice'].reshape(sales_nohighend['HousePrice'].shape[0],1)

# Create linear regression object
regr_noCCnohighend = linear_model.LinearRegression()

# Train the model using the training sets
regr_noCCnohighend.fit(sales_noCCnohighend_X_crimerate, sales_noCCnohighend_y_houseprice)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



### Do the coefficients change much?


```python
params_noCC_dict = {'interecpt':regr_noCC.intercept_[0],'CrimeRate':regr_noCC.coef_[0][0]}
pd.DataFrame(params_noCC_dict.items(), columns=['name','value'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>interecpt</td>
      <td>225233.551839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CrimeRate</td>
      <td>-2288.689430</td>
    </tr>
  </tbody>
</table>
</div>




```python
params_noCCnohighend_dict = {'interecpt':regr_noCCnohighend.intercept_[0],'CrimeRate':regr_noCCnohighend.coef_[0][0]}
pd.DataFrame(params_noCCnohighend_dict.items(), columns=['name','value'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>interecpt</td>
      <td>199098.852670</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CrimeRate</td>
      <td>-1838.562649</td>
    </tr>
  </tbody>
</table>
</div>



Above: We see that removing the outlying high-value neighborhoods has *some* effect on the fit, but not nearly as much as our high-leverage Center City datapoint.

*last edit 26/10/2016*

<a href="#top">Go to top</a>
