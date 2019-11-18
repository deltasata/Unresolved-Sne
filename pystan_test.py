import pystan
from scipy.optimize import curve_fit
import matplotlib
import numpy as np

def intrinsic_func(x,mu,dt):
    return mu*np.sin(x+dt)/np.sqrt(x+dt)

x=np.random.rand(200)*10;x=np.sort(x)
sigfx=0.2+np.random.rand(len(x))
print(np.amin(x), np.amax(x))


#mu and dt for the images
mu_l=[10.0,15.0,3.5,9.0]
dt_l=[0.0,1.57,0.5,-1.57]
#for simplicity just consider the 1st NI images for now
NI=2;  
fx=np.zeros((NI,len(x)))
for i in np.arange(NI): 
    fx[i]=intrinsic_func(x,mu_l[i],dt_l[i])
    
fx_total=np.sum(fx,axis=0)

y=np.random.normal(fx_total, sigfx) #the data


#pystan: modelling the intrinsic function with 9 parameters and interpolation
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
    real <lower=0.0, upper=2.0> dt[ni-1];#for simplicity just consider dt>0
    vector <lower=-1.0, upper=1.0> [NP] P;
    //real sig;
    
}
model {
    real dum;
    real td=0.5;
    real dum_t;
    int ti;
    real dtt;
    
    real mod_intr_f;
    
    for(i in 1:N) {
        dum=0;
        for (j in 1:ni){
            dum_t=x[i];
            if(j==1){dtt=0;}
            else {dtt=dt[j-1];}
            
            dum_t=dum_t+dtt;
            ti=1;
            //because there is no real to int conversion in stan we have to do it in this way
            while(dum_t>ti*td){
                ti=ti+1;
            }
            mod_intr_f=P[ti+1]+((P[ti+1]-P[ti])/td)*(dum_t-ti*td);
            dum+=mu[j]*mod_intr_f;
        }
        //y[i] ~ normal(dum, yerror[i]);
        target+=normal_lpdf(y[i]|dum, yerror[i]);
    }
}
"""
NP=25; # we see that the result is quite sensitive to this.
# Put our data in a dictionary
data = {'N': len(x),'ni':NI,'NP':NP, 'x': x, 'y': y, 'yerror':sigfx}

# Compile the model
sm = pystan.StanModel(model_code=model)

# Train the model and generate samples
fit = sm.sampling(data=data, iter=2000, chains=4, warmup=400, thin=3, seed=101)
print(fit)
