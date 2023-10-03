# ```BUIDL YOUR STOCHASTIC-LOGISTIC REGRESSION MODEL STEP-BY-STEP```
<img width="600px" height="250px" src="https://sebastianraschka.com/images/faq/logisticregr-neuralnet/schematic.png"></img>

#### <!-- AUTHOR -->Author : ```Dr. Amiehe-Essomba``` (c) 2023


## **```Objectives```**

*The main objective here is to implement a __Stochastic Logistic Regression__ model with a single neuron.* \
*To achieve this, we will use the sigmoid function(for binary classification) as the activation function for the neuron. It's* \
*essential to know that, there are several alternative activation functions available for use, including:* 

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

* __LeakyReLU function__ 
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
 * **X** : is the input matrix (n, m) that represents features. In this case n represents features and m samples

- ```python
  1. g(Z) = sigmoid(Z)
     is the true form of activation function that we are going to implement and use 

  2. loss = - ( (1-y) * log(1-g) + y * log(g) ) 
     is the loss function 

  3. loss = - (1/m) * ( (1-y) * log(1-g) + y * log(g) ).sum(axis=1) 
     is the cost function use to compute the gradient and optimize weights and biases 

  << This form of cost function is called **binary cross entropy** >>
    ```

## ```Backward Propagation Equations : ```

```python 
   
   Two equations are used to propagete the gradient (Eq.1 & Eq.2):  
   
   dw = dg * (dg/dz) * (dz/dw) = (1/m) *  X * ( g - a ).T    (Eq.1)
   db = dg * (dg/dz) * (dz/db) = (1/m).sum( g - a )          (Eq.2)

   dw is the weight derivative (dw = dcost/dw)
   db is the bias derivative (db = dcos/db)

   in general the true equations are defined as:

   dw = (dcost/da) * (da/dz) * (dz/dw)                      (Eq.3)
   db = (dcost/da) * (da/dz) * (dz/db)                      (Eq.4)

   with g(Z) = a                                            (Eq.5)

   By using equations (Eq.3, Eq.4, Eq.5) we can find the equations (Eq.1, Eq.2)

```


## __```Model schema representation:```__

```python
                                    Forward Propagation (w & b)
          +----------->------------->---------->----------->----------------------------+
          |                                                                             |             1 if p > 0.5 else 0           
INPUT( X * WEIGHT ) ----> LINEAR (Z = X.T * WEIGHT + BIAS) ----> ACTIVATION( SIGMOID(Z) ) ---> OUTPUT( Y[ idx([p, 1 - p]) ] )
          |                                                                              |                
          +------<-----------------<-----------------------<-------------------<---------+
                                    Backward Propagation (dw & db)

```

## __```Initialization :```__
```python
params = {'b' : np.zeros((ny, 1)), 'W' : np.zeros((ny, nx)) }

. nx is the features size of X matrix
. ny is the ouput size. In this case ny = 1 (True || False)
```
```For a single neuron we can initialise the bias to 0.0 and weight as np.zeros((ny, nx))```

## ```What we will do here :```
 - Building our deep neural network model(dnn) with one neuron using sigmoid function as activation function (LogisticRegression)
 - Loading data in the website (for binary classification : cat or not-cat)
 - Preprocessing data( **cleaning & normalizing**)
 - Training model
 - Testing model
 - Computing metrics(accuracy) to see how our model has performed
 
### __```Python Version :```__
   V >= 3.9.7

### __```NoteBook```__
   Take à look on [main.ipynb](https://github.com/amiehe-essomba/logisticRegression/blob/computer-vision/main.ipynb)

   I Hope that you will enjoy !!!
