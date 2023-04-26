import numpy as np

# Question 1 - The number of iterations it takes gauss-seidel to converge:

tol= 1e-6
A = np.array([[3, 1, 1],
              [1, 4, 1],
              [2, 3, 7]])
B = np.array([1, 3, 0])
r = np.array([0,0,0])
max_i = 50


def gaus_s (A, B, r, tol, max_i):
    num_i = 0
    while num_i < max_i:
      r_prev = np.copy(r)
      
      for j in range(len(B)):
        r[j] = (B[j] - np.dot(A[j, :j], r[:j]) - np.dot(A[j, j+1:], r_prev[j+1:])) / A[j, j]
    #Calculate Magnitude
      if np.linalg.norm(r - r_prev) < tol:
          return num_i + 1
      num_i += 1
    return -1

num_i = gaus_s(A, B, r, tol, max_i)

if num_i != -1:
    print(num_i)
else:
    print("Error")


# Question 3 - Determine the number of iterations necessary to solve f(x) = x3 - x2 + 2 = 0 using newton-raphson from the left side

#function 
def f(x):
    return x**3 - x**2 + 2
#derivative

def deriv(x):
    return 3*x**2 - 2*x

x = 0.5
tol = 1e-6
max_i = 100
converge = False
i = 0
while not converge and i < max_i:
    x_new = x - f(x) / deriv(x)
    if abs(x_new - x) < tol:
        converge = True
    x = x_new
    i += 1

print(i)
    
    
    
    
    
# Question 5 - The final value of the modified eulers method
def function(t: float , y: float):
    return y - (t**3)

def mod_euler_method(function, t0, y0, h, n):
    t = np.zeros(n+1)
    y = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in range(n):
        slo1 = function(t[i], y[i])
        t_m = t[i] + h / 2
        y_m = y[i] + (h / 2) * slo1
        slo2 = function(t_m, y_m)
        t[i+1] = t[i] + h
        y[i+1] = y[i] + h * slo2
    return y[-1]

t0 = 0.5
y0 = t0 ** 3
n = 100
a = 0
b = 3
h = (b - a) / n

final_val = mod_euler_method(function, t0, y0, h, n)
print("The final value of y using modified Euler's method is:", final_val)
