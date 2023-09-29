import numpy as np 

def generate_inputs_test(nx : int, ny : int = 1, samples : int = 100):

    X = np.random.randn(nx, samples)
    Y = np.random.randn(ny, samples)

    Y = (Y > 0.5) * 1
    return X, Y.astype('int')

def initialize_adam(params : dict = {}) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    params[f"W{l}"] = Wl
                    params[f"b{l}"] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v[f"dW{l+1}"] = ...
                    v[f"db{l+1}"] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s[f"dW{l+1}"] = ...
                    s[f"db{l+1}"] = ...

    """
    
    L = len(list( params.keys() )) // 2 # number of layers in the neural networks
    v = dict()
    s = dict()
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v[f"dW{l+1}"] = np.zeros(params[f"W{l+1}"].shape)
        v[f"db{l+1}"] = np.zeros(params[f"b{l+1}"].shape)
        s[f"dW{l+1}"] = np.zeros(params[f"W{l+1}"].shape)
        s[f"db{l+1}"] = np.zeros(params[f"b{l+1}"].shape)
    
    return v, s


def adam(
        params  : dict , 
        grads   : dict,
        v       : dict, 
        s       : dict, 
        t       : float, 
        learning_rate   : float = 0.01,
        beta1   : float = 0.9, 
        beta2   : float= 0.999,  
        epsilon : float = 1e-8
        ):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    params[f'W{l}'] = Wl
                    params[f'b{l}'] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads[f'dW{l}'] = dWl
                    grads[f'db{l}'] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    # number of layers in the neural networks
    layers      = len(list( params.keys())) // 2   
    # Initializing first moment estimate, python dictionary       
    v_corrected = dict()   
    # Initializing second moment estimate, python dictionary                     
    s_corrected = dict()                        
    
    # Perform Adam update on all parameters
    for l in range(layers):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v[f"dW{l+1}"] = beta1 * v[f"dW{l+1}"] + (1 - beta1) * grads[f"dW{l+1}"] 
        v[f"db{l+1}"] = beta1 * v[f"db{l+1}"] + (1 - beta1) * grads[f"db{l+1}"] 

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected[f"dW{l+1}"] = v[f"dW{l+1}"] / (1 - beta1**t)
        v_corrected[f"db{l+1}"] = v[f"db{l+1}"] / (1 - beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s[f"dW{l+1}"] = beta2 * s[f"dW{l+1}"] + (1 - beta2) * (grads[f"dW{l+1}"] ** 2)
        s[f"db{l+1}"] = beta2 * s[f"db{l+1}"] + (1 - beta2) * (grads[f"db{l+1}"] ** 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected[f"dW{l+1}"] = s[f"dW{l+1}"] / (1 - beta2 ** t)
        s_corrected[f"db{l+1}"] = s[f"db{l+1}"] / (1 - beta2 ** t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        params[f"W{l+1}"] = params[f"W{l+1}"] - learning_rate * v_corrected[f"dW{l+1}"] / np.sqrt(s_corrected[f"dW{l+1}"] + epsilon)
        params[f"b{l+1}"] = params[f"b{l+1}"] - learning_rate * v_corrected[f"db{l+1}"] / np.sqrt(s_corrected[f"db{l+1}"] + epsilon)

    return params, v, s