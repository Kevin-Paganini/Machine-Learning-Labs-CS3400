

class NumericalDifferentiation():
    
    def __init__(self, func, params, h):
        self.func = func
        self.params = params
        self.h = h
        
        
    def f(self):
        return self.func(self.params)
    
    
    def fp(self):
        gradients = []
        val_to_sub = self.func(self.params)
        
        for i in len(self.params):
            
            self.params[i] += self.h
            gradients.append(self.func(self.params[i]) - val_to_sub) / self.h
            self.params[i] -= self.h
        return gradients  