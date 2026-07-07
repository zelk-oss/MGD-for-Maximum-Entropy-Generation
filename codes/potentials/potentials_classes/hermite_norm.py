class Hermite_norm():
    def __init__(self, p,filters):
        super().__init__()
        self.p = p
        self.L2 = L2p_norm(1,filters)
        self.L4 = L2p_norm(2,filters)
        self.L6 = L2p_norm(3,filters)
        self.L8 = L2p_norm(4,filters)
        self.filters = filters
 
    def forward(self, x):
        if self.p == 1:
            x = self.L2(x)/2
        if self.p == 2:
            x = self.L4(x)/4-self.L2(x)*3/2
        if self.p == 3:
            x = self.L6(x)/6-self.L4(x)*10/4+self.L2(x)*15/2
        if self.p == 4:
            x = self.L8(x)/8- self.L6(x)*21/6+ self.L4(x)*105/4- self.L2(x)*105/2
            
        return x

    def grad(self, x, v=None, means=None):
        
        if self.p == 1:
            x = self.L2.grad(x, v,  means)/2
        if self.p == 2:
            x = self.L4.grad(x,v, means)/4-self.L2.grad(x, v, means)*3/2
        if self.p == 3:
            x = self.L6.grad(x, v, means)/6-self.L4.grad(x, v, means)*10/4+self.L2.grad(x,v, means)*15/2
        if self.p == 4:
            x = self.L8.grad(x, v, means)/8- self.L6.grad(x, v, means)*21/6+ self.L4.grad(x, v, means)*105/4- self.L2.grad(x, v, means)*105/2
        return x 
        
    def fit_micro(self,x):
        self.L2.fit_micro(x)
        self.L4.fit_micro(x)
        self.L6.fit_micro(x)
        self.L8.fit_micro(x)