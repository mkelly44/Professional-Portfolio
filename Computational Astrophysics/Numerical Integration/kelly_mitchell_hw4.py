import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxw

C = 3000.0
mat_den = 0.3
darkmat_den = 0.7
rad_den = 0.0

def r(z):
    return C/(np.sqrt( mat_den*(1+z)**3 + rad_den*(1+z)**2 + darkmat_den ))



### Rectangular Method ###



N_rec = 1000                                            # All N's for each method are the number of (rectangles, trapezoids, parabolas, etc) for each integration range a to b
a_rec = 0.0
b_rec = np.linspace(0.0, 10.0, num=40)
h_rec = (b_rec - a_rec)/N_rec                           # All h's are the tiny step sized used in each integration method

def I_rec(N_rec,h_rec,a_rec,b_rec):
    S = 0.0                                             # Variables inside fucntions are representations of sums that stay within that function!
    for k in range(0,N_rec):
        S += h_rec * r(a_rec + k * h_rec)               # For loop summation of rectangles 
    return(S)

rec_integrals = np.array([])
rec_integrals = np.append(rec_integrals, I_rec(N_rec,h_rec,a_rec,b_rec)/(1 + b_rec))             # Where integrals for each redshift range are stored



### Trapezoid Method ###



N_trap = 1000
a_trap = 0.0
b_trap = np.linspace(0.0, 10.0, num=40)
h_trap = (b_trap - a_trap)/N_trap

def I_trap(N_trap,h_trap,a_trap,b_trap):                            # Trapezoid Rule defined in class and from Newman
    S0 = 0.5*(r(a_trap) + r(b_trap))
    for k in range(1,N_trap):
        S0 += r(a_trap + k * h_trap)                            
    return(S0 * h_trap)


trap_integrals = np.array([])
trap_integrals = np.append(trap_integrals, I_trap(N_trap,h_trap,a_trap,b_trap)/(1 + b_trap))



### Simpson's Method ###



N_simp = 1000
a_simp = 0.0
b_simp = np.linspace(0.0, 10.0, num=40)
h_simp = (b_simp - a_simp)/N_simp

def I_simp(N_simp,h_simp,a_simp,b_simp):                    # Simpson Method formula with even/odd designation based upon asynchronous lecture formula
    S1 = r(a_simp) + r(b_simp)
    for k in range(1,N_simp,2):
        S1 += 4 * r(a_simp + k * h_simp)
    for k in range(1,N_simp,2):
        S1 += 2 * r(a_simp + k * h_simp)
    return( 1/3 * (h_simp * S1))
           
simp_integrals = np.array([])
simp_integrals = np.append(simp_integrals, I_simp(N_simp,h_simp,a_simp,b_simp)/(1 + b_simp))



### Gaussian Quadrature Method ###

z_quad = np.linspace(0.0, 10.0, num=40)

N = 1000
a_quad = 0.0
quad_integrals = np.array([])

for i in range(0,40):                                       # I could not figure out a cleaner way to code this because b_quad has dependence within the gaussxw(N) function from Newman, causing incorrect integral values 
    b_quad = 0.0                                            # I created a for loop for the entire simulation to run through all 40 integration ranges I wanted to plot at the end.
    b_quad += 0.25 * i                                  

    x,w = gaussxw(N)
    
    xprime = 0.5 * (b_quad - a_quad) * x + 0.5 * (b_quad + a_quad)                      # Formulation and coding from asynchronous lecture
    wprime = 0.5 * (b_quad - a_quad) * w
 
    def I_quad(N,xprime,wprime,b_quad):                              
        S2 = 0.0
        for k in range(0,N):
            S2 += wprime[k] * r(xprime[k])
        return(S2)

    storage = np.array([])                                                               # In each lap of the for loop the integral value is saved into a storage array and moved into another array where each of the 40 integral 
    storage = np.append(storage, I_quad(N,xprime,wprime,b_quad)/(1+b_quad))              # values are stored and then plotted
    quad_integrals = np.concatenate([quad_integrals, storage])



### np.trapz Method ###



b_trapz = np.linspace(0.0, 10.0, num=40)

trap_integrals = []
for n in range(1, 41):                                                                                      # For loop that begins by finding the necessary y values I am constructing trapzoids with
    y = []
    for k in range(0, n):                                              
        y = np.append(y, r(b_trapz[k]))                                                                     # Results of first for loop are stored into a y array
    trap_integrals = np.append(trap_integrals, np.trapz(y, b_trapz[0:n], 0.25) / (1 + b_trapz[n - 1]))       

                                                                                                            # Trapezoids with 0.25 z width going through all calculated y values added to array ready for plotting

### Plotting ###



plt.plot(b_rec, rec_integrals, label='Rectangular')
plt.plot(b_trap, trap_integrals, label='Trapezoidal')
plt.plot(b_simp, simp_integrals, label='Simpson')
plt.plot(z_quad, quad_integrals, label='Quadrature')
plt.plot(b_trapz, trap_integrals, label='np.trapz')
plt.ylabel('$D_{A}$')
plt.xlabel('z')
plt.title('Angular Size Over Various Redshift Ranges')
plt.legend(title='Integration Method')
plt.show()


