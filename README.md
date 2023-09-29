# ```BUIDL YOUR LOGISTIC REGRESSION MODEL STEP-BY-STEP```
<img width="600px" height="250px" src="https://sebastianraschka.com/images/faq/logisticregr-neuralnet/schematic.png"></img>

#### <!-- AUTHOR -->Author : ```Dr. Amiehe-Essomba``` (c) 2023


## **```Objectives```**

*The main objective here is to implement a __Logistic Regression__ model with a single neuron.* \
*To achieve this, we will use the sigmoid function as the activation function for the neuron. It's essential* \
*to recognize that there are several alternative activation functions available for use, including:*

* __tanh (hyperbolic tangent) function__
```python
f(x) = [np.exp(x) - np.exp(-x)] / [np.exp(x) + np.exp(-x)]
``` 

* __ReLU (Rectified Linear Unit) function__
```python
f(x) = np.maximum( 0, x )
```

* __identity or linear function__
```python
f(x) = x
```

* __sigmoid or logistic function__
```python
f(x) = 1 / (1 + np.exp(-x))
```

* __softmax function__ 
```python
f(xi) = np.exp(-xi) / np.sum( np.exp(-x), axis=0 )
```

* __LeakyRelU function__ 
```python
f(x) = np.maximum( 0.1, x )
```
 
## ```Forward Propagation Equations :```
 ```python
    Z = W.T.dot(X) + b
 ```

  *Where :*

 * **Z** : is the linearity
 * **W** : is the weight of neuron
 * **b** : is the bias  of neuron
 * **X** : is the input matrix (n, m) that represents the features. n is the features and m the samples

- ```python
  1. g(Z) = sigmoid(z)
     is the true form of activation function

  2. loss = - ( (1-y) * log(1-g) + y * log(g) ) 
     is the loss function

  3. loss = - (1/m) * ( (1-y) * log(1-g) + y * log(g) ).sum(axis=1) 
     is the cost function

  << binary cross entropy >>
    ```


## ```Backward Propagation Equations : ```

```python 
    dw = dg * (dg/dz) * (dz/dw) = (1/m) *  X * ( g - a).T
    db = dg * (dg/dz) * (dz/db) = (1/m).sum(g - a )
```


## __```Model schema representation:```__

```python
                                    Forward Propagation (WEIGHT & BIAS)
          +----------->------------->---------->----------->----------------------------+
          |                                                                             |             1 if p > 0.5 else 0           
INPUT( X * WEIGHT ) ----> LINEAR (Z = X * WEIGHT + BIAS) ----> ACTIVATION( SIGMOID(Z) ) ---> OUTPUT( Y[ idx([p, 1 - p]) ] )
          |                                                                              |                
          +------<-----------------<-----------------------<-------------------<---------+
                                    Backward Propagation (D_WEIGHT & D_BIAS)

```

## __```Initialization :```__
```python
params = {'b' : np.zeros((ny, 1)), 'W' : np.zeros((ny, nx)) }

. nx is the size of features in matrix X
. ny is the ouput size in this case ny = 1
```
```For a single neuron we can initialise the bias to 0.0 and weight to np.zeros((1, nx))```

## ```What we will do here :```
 - Building our dnn model (LogisticRegression)
 - Loading data from sklearn(cancer dataset)
 - Data preprocessing( **cleaning, normalization**)
 - Train model
 - Test model
 - Computing metrics to se how our model has performed
 
 ### __```Python Version :```__
   V >= 3.9.7
