import numpy as np
import math, copy
#Followed along lab in Stanford Course
#Lin Reg function 
# Y = mx + b
# Error/Loss = (y-yhat)^2/N
#Sample data set
x_train = np.array([1.0,2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x,y,w,b): #Compute the cost/loss function
    m = x.shape[0] #Amount of data points
    cost = 0

    for i in range(m):
        f_wb = w* x[i] + b
        cost += (f_wb - y[i])**2 
    total_cost = 1/(2*m)*cost

    return total_cost

def compute_gradient(x, y, w, b): 
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db #This only returns the gradient with respect to params w and b

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 

    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing


# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


#Sample X values
# x = np.random.randn(10,1) 

# y = 2*x + np.random.randn(10,1)
# #Parameters
# m = 0.0 #initialize as 0 first
# b = 0.0



# def descent(x, y, m, learning_rate): #Gradient descent function
#     dldm = 0.0 #Partial Deriv w respect to m
#     dldb = 0.0 #Partial Deriv w respect to b
#     N = x.shape[0]
#     #error/loss = (y - (mx+b))^2
#     for xi, yi in zip(x,y):
#         dldm = -2*xi*(yi-(m*xi+b)) #Partial deriving w/ respect to m
#         dldb = -2*(yi-(m*xi+b)) #Partial deriving w/respect to b

#     #Make an update to the m parameter
#     m = m - learning_rate*(1/N)*dldm

# learning_rate = 0.0
# for i in range(400):
#     pass