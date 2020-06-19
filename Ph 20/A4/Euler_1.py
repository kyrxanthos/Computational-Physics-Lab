"""
Ph20 Assignment 4
Created By Kyriacos Xanthos

"""

import numpy as np
from matplotlib import pyplot as plt



def traj_expl(x_0, v_0, h, N, meth):

    #ts = np.arange(0, h*(N), h)

    #perform the Euler method
    #starting with initial conditions
    xs = np.empty(N+1)
    xs[0] = x_0
    vs = np.empty(N+1)
    vs[0] = v_0
    if( meth == "explicid"):
        for i in range(N):
            xs[i+1] = xs[i] + h*vs[i]
            vs[i+1] = vs[i] - h*xs[i]
    elif( meth == "implicid"):
        for i in range(N):
            xs[i+1] = (xs[i] + h*vs[i])/(1+h**2 )
            vs[i+1] = (vs[i] - h*xs[i])/ (1+h**2)
            #xs[i+1] = (xs[i] + h*vs[i+1])
            #vs[i+1] = (vs[i] - h*xs[i+1])
    elif( meth == "sympletic"):
        for i in range(N):
            xs[i+1] = xs[i] + h*vs[i]/ (1+h**2 )
            vs[i+1] = vs[i] - h*xs[i+1]/ (1+h**2 )


    return xs, vs


def analytic(x_0, v_0, ts):
    xa = x_0*np.cos(ts) + v_0*np.sin(ts)
    va = -x_0*np.sin(ts) + v_0*np.cos(ts)
    
    return xa, va

def error(xs, vs, xa, va):
    x_error = (xs - xa)
    v_error = (vs - va)
    
    return x_error, v_error


def trunc_error(h,x_0, v_0, N, ts, meth):
    
    x_list = []
    h_list = []

    
    for n in range(0, 7):
        h_p = h/(2**n)
        x_max = max(traj_expl(x_0, v_0, h_p, N, meth)[0])
        x_list.append(x_max)
        h_list.append(h_p)
    
    return x_list, h_list

def energy(xs, vs, xa, va):
    Ene = xs**2 +vs**2
    A_Ene = xa**2 + va**2
    return Ene, A_Ene

def phase_space(x_0, v_0, h, N, xa, va):
    meth = "explicid"
    xs1, vs1 = traj_expl(x_0, v_0, h, N, meth)

    meth = "implicid"
    xs2, vs2 = traj_expl(x_0, v_0, h, N, meth)
    
    meth = "sympletic"
    xs3, vs3 = traj_expl(x_0, v_0, h, N, meth)
    
    #plt.plot(xs1, vs1, label = "Explicid")
    #plt.plot(xs2, vs2, color = "r", label = "Implicid")
    plt.plot(xs3, vs3, color = "g", label = "Sympletic")
    plt.plot(xa, va, color = "y", label = "Analytic")
    plt.xlabel("Velocity")
    plt.ylabel("Displacement")
    plt.title("Phase space")
    plt.legend()
    #plt.savefig('trial.png')    
    plt.show()
    


def all_plots(vs, xs, ts, xa, va, x_error, v_error, x_list, h_list, Ene, A_Ene, meth):
    

    
    plt.figure(figsize = (9.5, 4.1))
    plt.plot(ts, x_error, label = "displacement error")
    plt.plot(ts, v_error, color = "r", label = "velocity error")
    plt.plot(ts, Ene, color = "g", label = "Energy")
    plt.xlabel("Value of t")
    plt.ylabel("x error")
    plt.title("Error against time of Euler Symplectic method")
    plt.legend()
    #plt.savefig('trial.png')    
    plt.show()
    
    
        
    plt.figure(figsize = (9.5, 4.1))
    plt.plot(ts, xs, label = "x")
    plt.plot(ts, vs, color = "r", label ="v")
    plt.legend()
    plt.xlabel("Value of t")
    plt.ylabel("Value of x and v")
    plt.title("Approximate Solution with Euler’s Explicid Method")
    #plt.savefig('trial2.png')    
    plt.show()
        
    plt.figure(figsize = (9.5, 4.1))
    plt.plot(h_list, x_list, label = "displacement error")
    plt.xlabel("Value of step size")
    plt.ylabel("x error")
    plt.title("Truncation Error")
    plt.legend()
    #plt.savefig('trial.png')    
    plt.show()



    plt.figure(figsize = (9.5, 4.1))
    plt.plot(ts, Ene)
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy of System")
    #plt.savefig('trial.png')    
    plt.show()

    if (meth == "sympletic"):
        plt.figure(figsize = (9.5, 4.1))
        plt.plot(ts, xa, label = "analytic displacement")
        plt.plot(ts, xs, label = "sympletic displacement")
        plt.xlabel("time")
        plt.ylabel("displacement")
        plt.title("Comparison between analytic and symplectic method")
        plt.legend()
        #plt.savefig('trial.png')    
        plt.show()


def main():
    
    x_0 = 0
    v_0 = 5
    h = 0.5
    N = 1000
    ts = np.linspace(0, h*1.1*N, N + 1)
    meth = "explicid"

    
    xs, vs = traj_expl(x_0, v_0, h, N, meth)
    xa, va = analytic(x_0, v_0, ts)
    x_error, v_error = error(xs, vs, xa, va)
    Ene, A_Ene = energy(xs, vs, xa, va)
    x_list, h_list = trunc_error(h,x_0, v_0, N, ts, meth)
    all_plots(vs, xs, ts, xa, va, x_error, v_error, x_list, h_list, Ene, A_Ene, meth)
    x_list, h_list = trunc_error(h,x_0, v_0, N, ts, meth)
    phase_space(x_0, v_0, h, N, xa, va)


main()


