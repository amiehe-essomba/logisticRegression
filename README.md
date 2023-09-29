# <!-- TITLE -->TITLE : ```BUIDL YOUR LOGISTIC REGRESSION MODEL STEP-BY-STEP```
<img width="600px" height="250px" src="https://sebastianraschka.com/images/faq/logisticregr-neuralnet/schematic.png"></img>

#### <!-- AUTHOR -->Author : ```Dr. Amiehe-Essomba``` (c) 2023


## **```Objectives```**

*Our objective is to implement a __Logistic Regression__ model with a single neuron.* \
*To achieve this, we will utilize the sigmoid function as the activation function. It's essential* \
*to recognize that there are several alternative activation functions available for use, including:*

* __than__
```python
f(x) = [np.exp(x) - np.exp(-x)] / [np.exp(x) + np.exp(-x)]
``` 

* __ReLU (Rectified Linear Unit)__
```python
f(x) = max(X, 0 )
```

* __identity__
```python
f(x) = x
```

* __sigmoi__
```python
f(x) = 1 / (1 + np.exp(-x))
```

* __softmax__ 
```python
f(xi) = np.exp(-xi) / np.sum( np.exp(-x) )
```

 
## ```Feed Forward Propagation Equations :```
 ```python
    Z = W.T.dot(X) + b
 ```

  *Where :*

 * **Z** : is a parameter
 * **W** : is the weight 
 * **b** : is the bias 
 * **X** : a matrix (n, m) that represents the features or DNN inputs 

- ```python
  1. g(Z) = (1 / (1 + exp(-Z))) 
     is the true form of activation function
  2. loss = - ( (1-y) * log(1-g) + y * log(g) ) 
     is the loss function
  3. loss = - (1/m) * ( (1-y) * log(1-g) + y * log(g) ).sum() 
     is the cost function
  << binary cross entropy >> usually used for binary classification issues>>
  
  4. L = (1/m) * sum(loss) 
     is the cost function use to compute the gradient 
  5. grad(O) = dL/dO 
     is the gradient form

  with O = (W, b)

  6. dg    = dL/dg = (g-y)/(1-g) 
     the derivative of L in function of g
  7. dg/dz = (1-g) 
     the derivative of g in function of Z

  8. dz/db = 1 
     the derivative of Z in function of b
  9. dz/dw = X 
     the derivative of Z in function of W
    ```

## ```Back Forward Propagation Equations : ```

```python 
    dw = dg * (dg/dz) * (dz/dw) = (1/m) *  X * ( g - a).T
    db = dg * (dg/dz) * (dz/db) = (1/m).sum(g - a )
    d_epsilon = loss[l] - loss[l-1]
```

## ```Turning Hyperparameters :```

* **learning_rate** 
* **number of layers** in this case just 1 layer (NL = 1) with one neurone
* shape is the dimension in features
* **max_iter** for iterations to reach to the convergence
* **epsilon** 
We break the loop when (dgrad < epsilon) where epsilon is the tolerence \
in this case we'll use ```while loop``` to be sure that we reach to the convergence (local minimum is found)

* **shape** = X.shape = (n, m)

to build a DNN with a single neurone as below we need to initialize W and b to zeros 

## __```Model representation :```__

```python
                                    Feed Forward Propagation
          +----------->------------->---------->----------->----------------------------+
          |                                                                             |                       
INPUT( X * WEIGHT ) ----> LINEAR (Z = X * WEIGHT + BIAS) ----> ACTIVATION( SIGMOID(Z) ) ---> OUTPUT( Y[ idx([p, 1 - p]) ] )
          |                                                                              |                
          +------<-----------------<-----------------------<-------------------<---------+
                                    Back Forward Propagation

```

## __```Initialization :```__
```python
params = {'b' : np.zeros((NL, 1)), 'W' : np.zeros((1, n)) }
```

Then, after initaliazing parameters we can now compute the rest of functions

```python
index, costs = 0, []
while d_epsilon > epsilon:
    Z       = params["W"].dot(X) + params['b']
    g       = (1.0 / (1.0 + np.exp(-Z)))
    loss    = - ( (1-y) * log(1-g) + y * log(g) )
    cost    = - (1/m) * ( (1-y) * log(1-g) + y * log(g) ).sum(axis=1)
    costs.append(cost)
    # back propagation 
    params['W'] = params['W'] - learning_rate * dgrad/dW
    params['b'] = params['b'] - learning_rate * dgrad/db

    if index == 0: pass 
    else:  d_epsilon = costs[index] - costs[index-1]

    index += 1
```

## ```What we will do here :```

 - Loading data from sklearn(cancer dataset)
 - Data preprocessing( **cleaning, normalization**)
 - Building DNN model
 - Train model
 - Test model
 - Computing metrics for evaluating models
 
 ### __```Python Version :```__
   V >= 3.9.7
