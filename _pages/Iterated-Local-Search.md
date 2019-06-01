---
permalink: /Iterated-Local-Search/
header:
  image: "/images/digital-transition2.jpg"
---

# Iterated Local Search

This algorithm has resemblances with algorithm 1. However, instead
of starting each iteration from a completely random solution, the starting solution is a
‘perturbation’ of the current solution. This perturbation is generally stronger than the
operator used for hill-climbing. For example, you can use a 2-flip or 3-flip operator for a
binary representation, when your hill-climbing uses 1-flip moves. Again you have a
parameter num_iter, which determines how many perturbations you will attempt. You
need to keep record of the best solution found. This will be the output of the algorithm.


```python
import urllib2  # the lib that handles the url stuff
import numpy as np
import pandas as pd
#from random import randint
import random

input_data = []
url = "http://www.cs.stir.ac.uk/~goc/source/hard200.txt"
data = urllib2.urlopen(url) # it's a file like object and works just like a file
for line in data: # files are iterable
    input_data.append(map(int,line.split()))

instance_number = input_data.pop(0)[0]
max_capacity = input_data.pop()[0]
df = pd.DataFrame(input_data, columns=['no.', 'value', 'weight'])
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no.</th>
      <th>value</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>835</td>
      <td>735</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1670</td>
      <td>1470</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3340</td>
      <td>2940</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1087</td>
      <td>987</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1087</td>
      <td>987</td>
    </tr>
  </tbody>
</table>
</div>




```python
#function to generate a binary representation of the items selected.
def binrep(n,r):
    return "{0:0{1}b}".format(n, r)
#random.randint(1, 2**instance_number)
weights = np.array(df["weight"])   
values = np.array(df["value"])
```


```python
best_value = 0
num_inter = 1000 #how many random points we use to initially use for search
n_flip = 2
lst_best_value = []
items_selected = np.array(map(int,binrep(random.randint(1, 2**instance_number), instance_number)))
while num_inter >0:
    max_eval = 1000
    for i in range(0,n_flip):
        idx = random.randint(0,instance_number-1)
        items_selected[idx] = int(not items_selected[idx])
    while max_eval >0:
        total_value = np.dot(values,items_selected)
        total_weight = np.dot(weights,items_selected)
        if total_weight <= max_capacity:
            if total_value > best_value:
                best_value = total_value
                lst_best_value.append(best_value)
        max_eval -=1
    num_inter -= 1
print lst_best_value
print "best value = ",max(lst_best_value)
```

    [124386, 125272, 127014, 127404, 130015, 131189, 131698, 131735]
    best value =  131735



```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(list(xrange(0,len(lst_best_value))),lst_best_value)
plt.show
```




    <function matplotlib.pyplot.show>




![png](/images/Iterated-Local-Search/output_5_1.png)


*last editted: 01/06/19*
