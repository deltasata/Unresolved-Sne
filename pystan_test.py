import pystan
from scipy.optimize import curve_fit
import matplotlib
import numpy as np


def intrinsic_func(x,mu,dt):
    return mu*np.sin(x+dt)

x=np.random.rand(200)*6;x=np.sort(x)
sigfx=0.2+np.random.rand(len(x))

mu_l=[10.0,4.0,3.5]
dt_l=[0.0,1.57,0.5]
NI=2; 
fx=np.zeros((NI,len(x)))
for i in np.arange(NI): 
    fx[i]=intrinsic_func(x,mu_l[i],dt_l[i])
    
fx_total=np.sum(fx,axis=0)

y=np.random.normal(fx_total, sigfx)


model = """
data {
    int<lower=0> N;
    int<lower=0> ni;
    int<lower=5> NP;
    vector[N] x;
    vector[N] y;
    vector[N] yerror;
}
parameters {
    real<lower=1.0, upper=20> mu [ni];
    real <lower=0, upper=2> dt[ni-1];
    vector <lower=-1, upper=1> [NP] P;
    //real sig;
    
}
model {
    real dum;
    vector[NP] tm;
    real td=0.6;
    real dum_t;
    int ti;
    real dtt;
    
    real mod_intr_f;
    
    for(i in 1:N) {
        dum_t=x[i];
        dum=0;
        for (j in 1:ni){
        
            dtt=dt[1];
            if(j==0){dtt=0;}
            dum_t=dum_t+dtt;
            ti=1;
            while(dum_t<ti*td){ti=ti+1;}
            mod_intr_f=P[ti+1]+((P[ti+1]-P[ti])/td)*(dum_t-ti*td);
            dum+=mu[j]*mod_intr_f;
        }
        y[i] ~ normal(dum, yerror[i]);
        
    }
}
"""
NP=15;
# Put our data in a dictionary
data = {'N': len(x),'ni':NI,'NP':NP, 'x': x, 'y': y, 'yerror':sigfx}

# Compile the model
sm = pystan.StanModel(model_code=model)

# Train the model and generate samples
fit = sm.sampling(data=data, iter=2000, chains=4, warmup=400, thin=3, seed=101)
print(fit)
