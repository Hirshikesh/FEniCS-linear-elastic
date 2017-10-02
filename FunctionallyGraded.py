from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# mesh 
mesh = Mesh('PlCr.xml')
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)
print mesh.hmin()
plot(mesh)
print mesh.num_cells()
interactive()

# Material definition
x = SpatialCoordinate(mesh)
Eo = 10000
Ew = 50000
alpha = (1.0/10.0)*ln(Ew/Eo)
print alpha
#beta = (1.0/30.0)*ln(Eh/E0)

E = Expression('Eo*exp(alpha*x[0])', alpha=alpha, Eo = Eo)
nu = 0.25
#lmbda = Expression('(E*nu/((1.0 + nu )*(1.0-2.0*nu)))', E = E, nu = nu)
#mu = Expression('(E/(2*(1+nu)))', E = E, nu= nu)
# functionSpace
lmbda = E*nu/((1.0 + nu )*(1.0-2.0*nu))

mu = E/(2*(1+nu))
AA = np.linspace(0,10,100)
'''
for (b,c) in enumerate(AA):
    CC = (c,0.0)
    #print c
    #print lmbda(CC)
    plt.scatter(c,E(CC))
plt.show()
for (b,c) in enumerate(AA):
    CC = (c,0.0)
    #print lmbda(CC)
    plt.scatter(c,mu(CC))
plt.show()
'''

V = VectorFunctionSpace(mesh, 'CG',1)
u,v = TrialFunction(V), TestFunction(V)

class top(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-5
        return abs(x[1]-15.0) < tol 

class bottom(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-5
        return abs(x[1]) < tol and x[0] >= 4.0
class bottomR(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-5
        return abs(x[1]) < tol and x[0] >= 10.0  

Bottom = bottom()
BottomR = bottomR()
Top = top()

bc1=DirichletBC(V.sub(1),Constant(0.0),Bottom)
bc2=DirichletBC(V.sub(0),Constant(0.0),BottomR, method = 'pointwise')
bc = [ bc1, bc2]

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Top.mark(boundaries, 1)
Bottom.mark(boundaries, 2)
BottomR.mark(boundaries, 3)

# Define new measures associated with the exterior boundaries
ds = ds(domain=mesh , subdomain_data=boundaries)
# Boundary conditions - Neumann (stress on L&R side)
gR = Constant((0.0, 1.0))


def epsilon(v_disp):
    #return 0.5*(grad(v_disp)  + grad(v_disp).T)
    return sym(grad(v_disp))

def sigma(v_disp):
    return 2.0*mu*epsilon(v_disp) + lmbda*tr(epsilon(v_disp))*Identity(2)

a = inner(grad(v),sigma(u))*dx
L = inner(gR, v)*ds(1)
u = Function(V)
problem = LinearVariationalProblem(a, L, u, bc)
solver = LinearVariationalSolver(problem)
solver.solve()
u1, u2 = split(u)
plot(u)
plot(u1)
plot(u2)
interactive()

ZZ = np.matrix([[0.030930169368716176, 0.0011196721311475407],
                [0.52576759351508, 0.0010327868852459013],
                [1.0162802282401953, 0.0009393442622950816],
                [1.519178516438728, 0.0008442622950819669],
                [2.0136536545602755, 0.000744262295081967],
                [2.516053799474686, 0.0006311475409836064],
                [2.771284303957975, 0.0005704918032786884],
                [3.022235304773119, 0.000504918032786885],
                [3.215446970383118, 0.00044918032786885227],
                [3.359116022099448, 0.00039999999999999986],
                [3.5151707272891954, 0.00034918032786885234],
                [3.658477492980709, 0.0002868852459016393],
                [3.760710080608641, 0.00023770491803278677],
                [3.8586178788153243, 0.00018196721311475404],
                [3.9114889955619976, 0.00014590163934426214],
                [3.9600353228874203, 0.00010327868852459004],
                [4.011095009510009, 0.0000016393442622949974] ])
x=ZZ[:,0]
y=ZZ[:,1]
plt.plot(x,y, 'ro', label = 'Ref')
plt.hold(True)

AA = np.linspace(0,4,100)
print AA
for (b,c) in enumerate(AA):
    CC = (c,0.0)
    print u2(CC)
    plt.scatter(c,u2(CC))
plt.xlabel('x')
plt.ylabel('Crack displacement u2')
plt.show()

