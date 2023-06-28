import numpy as np
#Lin Reg function 
# Y = mx + b
# Error/Loss = (y-yhat)^2/N

#Sample X values
x = np.random.randn(10,1) 

y = 2*x + np.random.randn(10,1)
#Parameters
m = 0.0 #initialize as 0 first
b = 0.0



def descent(x, y, m, learning_rate): #Gradient descent function
    dldm = 0.0 #Partial Deriv w respect to m
    dldb = 0.0 #Partial Deriv w respect to b
    N = x.shape[0]
    #error/loss = (y - (mx+b))^2
    for xi, yi in zip(x,y):
        dldm = -2*xi*(yi-(m*xi+b)) #Partial deriving w/ respect to m
        dldb = -2*(yi-(m*xi+b)) #Partial deriving w/respect to b

    #Make an update to the m parameter
    m = m - learning_rate*(1/N)*dldm

learning_rate = 0.0
for i in range(400):
    pass