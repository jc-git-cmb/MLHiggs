import tensorflow as tf
import tensorflow.compat.v1 as v1
import tensorflow.keras as keras 
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Layer

### Determine GMM 

#determine if eager execution is occuring 
print('Eager exc',tf.executing_eagerly())
print('tensorflow: %s' % tf.__version__)

#define a function that computes the CDF (Cumulative Distribution Function) for the normalised Gaussain 
#input = i to function - a Tensor 
#use tensorflow.erf to return erf function - outputs a Tensor of same type as the input

def CDF(i):
    CDF = 0.5 * (1. + tf.math.erf(i / np.sqrt(2.)))
    return CDF

def integral_gauss(mean, sigma):

    #dot makes 0 and 1 a float 
    i0 = (0. - mean) / sigma
    i1 = (1. - mean) / sigma

    integ = CDF(i1) - CDF(i0)

    #use the Keras backend to evaluate the integral over the range 0 - 1 
    if K == np:
        integ = K.eval(integ)

    return integ

#create a function to calculate the normalised gaussian function (PDF) for each input value i 
#coeff = normalisation factors for the Gaussian functions 
#mean and sigma are the means and sigma values for the unit Gaussian function 
def gauss( i, coeff, mean, sigma):
    gauss = coeff * K.exp( - K.square(i - mean) / 2. / K.square(sigma)) / K.sqrt( 2. * np.pi * K.square(sigma) )
    return gauss

#define a function to calculate the Gaussian Mixture Model (GMM) for the ANN PDF 
#n is the number of Gaussians included in the GMM
def GMM( i, coeffs, means, sigma, n):

    #create tensor containing 0's to be added to
    #this is the posterior pdf values 
    gmm = K.zeros_like(i)

    for j in range(n):
        #determine the normalised gaussian function for all rows and each column in coeffs, means and sigma using keras backend 
        #each column will correspond to a gaussian function 
        comp  = gauss( i, coeffs[:,j], means[:,j], sigma[:,j])
        #divide the gaussian by the integral over the unit range i.e. 0-1 
        comp2 = comp / integral_gauss(means[:,j], sigma[:,j])
        #append this value to gmm (the posterior pdf)
        gmm += comp2

    return gmm


#Create Gradiant Reversal Layer and Posterior Layer for ANN

NUM_GRADIENT_REVERSALS=0

def ReverseGradient (lambda_val):
    #Reverse the incoming gradient's sign whilst training and multiply by lambda parameter 
    def rev_grad_func (x, lambda_val=lambda_val):
        #defined the number of gradient reversals as a global variable that can be altered in this function
        global NUM_GRADIENT_REVERSALS

        #define the gradient reversal name and format it such that the number of gradient reversals at that moment are depicted 
        gradient_name = "GradientReversal{}".format(NUM_GRADIENT_REVERSALS)

        #Add 1 to gradient reversals each time the function is 
        NUM_GRADIENT_REVERSALS += 1

        #decorator, create a new tensorflow op:
        # if op has m inputs and n outputs, gradient takes op and n outputs (= gradients wrt each output of op - as Tensor) 
        # returns an m tensor object (partial gradients wrt each input of op)
        @tf.RegisterGradient(gradient_name)

        #function to reverse the gradient and multiply it by lambda 
        def _rev_grad(op, gradient):
            return [tf.negative(gradient) * lambda_val]

        #retrieves tensorflow session , or returns a gobal session, or creates a global session
        #tensorflow operation represented as a dataflow graph and execute 
        
        g = v1.keras.backend.get_session().graph

        
        #g = tf.compat.v1.keras.backend.get_session().graph
        #gradient override = A context manager that sets the alternative op type to be used for one or more ops created in that context.
        with g.gradient_override_map({'Identity': gradient_name}):
            #retruns a tensor of same shape as x
            y = tf.identity(x)
    
        return y

    return rev_grad_func


#define a class for the Gradient Reversal Layer for the Combined Model
class GradientReversalLayer (Layer):

    #initialise 
    def __init__ (self, lambda_val, **kwargs):
        
        #constructor for the base class
        #Return a proxy object that delegates method calls to a parent or sibling class of type.
        #initialises the class as the parent class 
        super(GradientReversalLayer, self).__init__(**kwargs)

        # Define variables 
        self.supports_masking = False
        self.lambda_val = lambda_val
        self.gr_op = ReverseGradient(self.lambda_val)

    #return the gradient reversal function 
    def call (self, x, mask=None):
        return self.gr_op(x)

    #define output shape of the layer as same as the input 
    def compute_output_shape (self, input_shape):
        return input_shape


#define a class for the Posterior Layer for the Adversary Model
class PosteriorLayer (Layer):

    def __init__ (self, n, **kwargs):
        
        #constructor for the base class
        #Return a proxy object that delegates method calls to a parent or sibling class of type.
        #initialises the class as the parent class 
        super(PosteriorLayer, self).__init__(**kwargs)

        # Define Variable - number of Gaussians in the misture model 
        self.n = n

    #determine the pdf values by calling the GMM function 
    #requires Keras backend for back-propagation functionality
    def call (self, x, mask=None):
       
        # inputs to GMM 
        coeffs, means, sigma, m = x

        # Compute the pdf from the GMM
        #feed the function the input values, normalisation coefficients, means and sigmas for the n Gaussian functions
        pdf = GMM(m[:,0], coeffs, means, sigma, self.n )

        #returns outputs of shape of invididual input shapes multiplied together
        return K.flatten(pdf)

    #output shape of layer
    def compute_output_shape (self, input_shape):
        return (input_shape[0][0], 1)
