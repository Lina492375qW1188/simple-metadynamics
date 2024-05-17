import numpy as np

class double_well:
    
    def __init__(self):
        """
        param['B'] = B: height of barrier;
        param['x0'] = x0: minima of double well.
        """
        self.param={}
        
    def act(self, x):
        """
        defining an act function taking simulation variable as input argument.
        x: array-like.
        """
        dx = x**2 - self.param['x0']
        return self.param['B']*dx**2
    
    def deriv(self, x):
        """
        derivative.
        """
        return 4 * self.param['B'] * x**3 - 4 * self.param['B'] * self.param['x0'] * x


class harmonic:
    
    def __init__(self):
        self.param={}
        
    def act(self, x):
        """
        defining an act function taking simulation variable as input argument.
        x: array-like.
        """
        dx = x - self.param['x0']
        return self.param['K'] * dx**2
    
    def deriv(self, x):
        """
        derivative.
        """
        return 2 * self.param['K'] * (x - self.param['x0'])
        
        
class simple_md:
    
    def __init__(self, seed=0):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.external_pe = []
        
    def set_param(self, dt, kT=1.0, damping=1.0):
        self.dt = dt
        self.kT = kT
        self.damping = damping
        
    def set_init_x(self, init_x, init_v=0.0):
        self.init_x = init_x
        self.init_v = init_v
        self.traj = [[0, self.init_x, self.init_v]]
        
    def dvdt_f(self, x, v):
    
        grad = 0
        for pe in self.external_pe:
            grad += pe.deriv(x)

        return - grad - self.damping * v
    
    def xi(self, kT, damping):
        """
        Wiener noise.
        """
        sigma = np.sqrt( 2 * kT * damping )
        return np.random.normal(loc=0.0, scale=sigma)
    
    def run(self, nsteps):
        """
        nsteps: number of steps.
        """
        for i in range(int(nsteps)):
            
            xi = self.xi(self.kT, self.damping)
            
            x = self.traj[-1][1]
            v = self.traj[-1][2]

            predict_x = x + self.dt * v
            predict_v = v + self.dt * self.dvdt_f(x,v) \
                          + xi * np.sqrt(self.dt)

            xf = x + 0.5 * self.dt * (predict_v + v)
            vf = v + 0.5 * self.dt * (self.dvdt_f(predict_x,predict_v) + self.dvdt_f(x,v)) \
                   + xi * np.sqrt(self.dt)

            self.traj.append([i, xf, vf])
        

class simple_metad:
    
    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)
        self.external_pe = []
        
    def set_param(self, dt, kT=1.0, damping=1.0):
        self.dt = dt
        self.kT = kT
        self.damping = damping
        
    def set_init_x(self, init_x, init_v=0.0):
        self.init_x = init_x
        self.init_v = init_v
        self.traj = [[0, self.init_x, self.init_v]]
        self.vbias = []
        
    def set_metad(self, init_h, sigma, stride, bias_factor):
        self.init_h = init_h
        self.sigma = sigma
        self.stride = stride
        self.bias_factor = bias_factor
        self.dT = (self.bias_factor-1)*self.kT
        
        self.h_t = [self.init_h]
        self.x_t = [self.init_x]
        
    def dvdt_f(self, x, v):
    
        grad = 0
        for pe in self.external_pe:        
            grad += pe.deriv(x)

        return - grad - self.damping * v
    
    def xi(self, kT, damping):
        """
        Wiener noise.
        """
        sigma = np.sqrt( 2 * kT * damping )
        return np.random.normal(loc=0.0, scale=sigma)
    
    def gauss_bias(self, x, h_t, x_t, sigma=1.0):
    
        h_t = np.array(h_t)
        x_t = np.array(x_t)

        dx = x - x_t
        gauss_e = h_t * np.exp(-0.5 * dx**2 / sigma**2)
        gauss_f = gauss_e * dx / sigma**2

        return np.sum(gauss_e), np.sum(gauss_f)
    
    def run(self, nsteps):
        """
        nsteps: number of steps.
        """
        for i in range(int(nsteps)):
            
            xi = self.xi(self.kT, self.damping)
            
            x = self.traj[-1][1]
            v = self.traj[-1][2]
            
            bias_e, bias_f = self.gauss_bias(x, 
                                        self.h_t, 
                                        self.x_t, 
                                        self.sigma)

            predict_x = x + self.dt * v
            predict_v = v + self.dt * self.dvdt_f(x,v) \
                          + self.dt * bias_f \
                          + xi * np.sqrt(self.dt)
            
            bias_e_pred, bias_f_pred = self.gauss_bias(predict_x, 
                                                  self.h_t, 
                                                  self.x_t, 
                                                  self.sigma)
            
            if i%self.stride==0:
                self.x_t.append(x)
                h = self.init_h * np.exp( -bias_e / self.dT )
                self.h_t.append(h)

            xf = x + 0.5 * self.dt * (predict_v + v)
            vf = v + 0.5 * self.dt * (self.dvdt_f(predict_x,predict_v) + self.dvdt_f(x,v)) \
                   + 0.5 * self.dt * (bias_f_pred + bias_f) \
                   + xi * np.sqrt(self.dt)

            self.traj.append([i, xf, vf])
            self.vbias.append(0.5*(bias_e + bias_e_pred))
        
        
