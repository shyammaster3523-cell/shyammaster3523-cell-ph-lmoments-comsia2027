import numpy as np, json
from phlm import *
# theta = (log l1, log l2, log l3, a1, a2), a3 = 1-a1-a2 ; output = log lambda_1..lambda_5 (lambda_r>0 here)
def G(th):
    l=np.exp(th[:3]); a=[th[3],th[4],1-th[3]-th[4]]
    al,S=cf1(l,a); return ph_lmoments(al,S,5,"dense")
def J(th,h=1e-6):
    g0=G(th); Jm=np.zeros((5,5))
    for k in range(5):
        e=np.zeros(5); e[k]=h
        Jm[:,k]=(G(th+e)-G(th-e))/(2*h)
    # relative sensitivity: d log lambda / d theta
    return Jm/g0[:,None]
