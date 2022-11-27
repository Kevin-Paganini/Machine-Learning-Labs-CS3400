import numpy as np
import copy

class Optimizer:
    """
    Implements Gradient Descent using numerical differentiation for calculating the gradient.
    """
    def __init__(self, step_size, max_iter, tol, delta):
        """
        Max_iter -- maximum number of iterations to run
        step_size -- also known as lambda
        tol --
        delta -- perturbation to use in numerical differentiation
        """
        self.max_iter = max_iter
        self.step_size = step_size
        self.delta = delta
        self.tol = tol
        
    
    def optimize(self, cost_func, starting_params):
        """
        Finds parameters that optimize the given cost function.
        
        This method should implement your iterative algorithm for updating your parameter estimates.
        Use an updated estimate of the gradient to update the parametes.
        
        Give consideration for what the exit conditions of this loop should be.
        
        Returns a tuple of (optimized_param, iters)
        """
        total_iterations = 0
        params = starting_params
        for iteration in range(self.max_iter):
            total_iterations = iteration
            
            gradient = self._gradient(cost_func, params)
            
            
            new_params = self._update(params, gradient)
            params = new_params
            change = self.step_size * gradient
            
            if all(np.abs(change) < self.tol):
                return params, total_iterations
        print(f'Found params: {params}, total_iterations: {total_iterations}')
        return params, total_iterations
                
    
    def optimize_with_intermediate_results(self, cost_func, starting_params):
        """
        Finds parameters that optimize the given cost function.
        
        This method should implement your iterative algorithm for updating your parameter estimates.
        Use an updated estimate of the gradient to update the parametes.
        
        Give consideration for what the exit conditions of this loop should be.
        
        Returns a tuple of (optimized_param, iters, intermediate_results)
        """
        
        total_iterations = 0
        params = starting_params
        intermediate_results = [starting_params]
        for iteration in range(self.max_iter):
            total_iterations = iteration
            
            gradient = self._gradient(cost_func, params)
            
            
            new_params = self._update(params, gradient)
            intermediate_results.append(new_params)
            params = new_params
            change = self.step_size * gradient
            
            if all(np.abs(change) < self.tol):
                return params, total_iterations, intermediate_results
        print(f'Found params: {params}, total_iterations: {total_iterations}')
        return (params, total_iterations, intermediate_results)
    
    def _calculate_change(self, old, new):
        """
        Calculates the change between the old and new parameters.
        Returns a scalar.
        """
        print(f'Calculate change: {np.abs(new - old)}')
        return np.abs(new - old)
        
    
    def _gradient(self, cost_func, params):
        """
        Numerically estimates the gradient (first derivative) of the cost function
        at param.
        
        First-order numerical differentiation
        df/dx = [ f(x + delta) - f(x) ] / delta
        
        Should return the gradient at the caluclated point
        """
        gradients = []
        new_params = copy.deepcopy(params)
        for idx in range(len(params)):
            new_params[idx] = new_params[idx] + self.delta
            diff = cost_func.cost(new_params) - cost_func.cost(params)
            div = diff / self.delta
            gradients.append(div)
            new_params[idx] = new_params[idx] - self.delta
            
        
        return np.array(gradients).reshape(-1,)
        
            
    def _update(self, param, gradient):
        """
        Updates the param vector using the Gradient Descent algorithm.                
        
        Returns the new parameters.  (Do not modify input)
        """
        return param - self.step_size * gradient
        