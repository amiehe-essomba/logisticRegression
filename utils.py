import numpy as np 
import h5py
import matplotlib.pyplot as plt

def url_img_read( url : str, show_img : str = False):
    from PIL import Image
    import requests
    from io import BytesIO
    import numpy as np 
    import time

    image = None
    # Replace 'url' with the URL of the image you want to read

    try:
        start = time.time()
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Read the image from the response content
            image_data = BytesIO(response.content)
            image = Image.open(image_data) #Image.open(image_data)

            # You can now work with the 'image' object (e.g., display or process it)
            # For example, you can display the image:
            image = np.array(image).astype(np.float32) / 255 
            if show_img : 
                plt.figure(figsize=(2, 2))
                plt.imshow(image, interpolation="nearest", cmap="plasma")
                plt.axis('off')
                plt.show()
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    end = time.time()

    print(f"response time : {np.round( end-start, 4 )}s")

    return image

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

def load_dataset(path : str  ='./datasets'):
    train_dataset = h5py.File(f'{path}/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(f'{path}/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

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