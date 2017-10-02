from fenics import *


mesh = UnitSquareMesh(2,2)
ndim = mesh.geometry().dim()
print "number of unknown",mesh.num_vertices()
print "number of elements",mesh.num_cells()
plot(mesh, interactive = True)
V = VectorFunctionSpace(mesh,"CG",1)
E = 1.0
nu = 0.3
print E, nu

#E = (E*(1.0+2.0*nu) )/((1.0+nu)**2)
E = E/(1.0-nu**2)
print E
nu = nu/(1.0- nu)
print nu


lmbda = E*nu/((1.0 + nu )*(1.0-2.0*nu))
mu = E/(2.*(1.+nu))

u = TrialFunction(V)
v = TestFunction(V)

class Left(SubDomain): 
    def inside(self,x,on_boundary):
       return (near (x[0],0.0) and on_boundary)
class  Right(SubDomain):
   def  inside(self , x, on_boundary ):
       return (near (x[0],1.0) and on_boundary)
  
right   = Right()
left = Left()

bc_fixed = DirichletBC(V, Constant ((0.0,0.0)),  left)
disp_right = DirichletBC(V.sub(0), Constant(0.1), right)
bcs = [bc_fixed, disp_right]

def  epsilon(v):
    return  sym(grad(v))
def  sigma(v):
    return 2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity (ndim)
a = inner(grad(v), sigma(u))*dx
u = Function(V)
solve(lhs(a)==rhs(a),u,bcs)
(u1, u2) = u.split(True)
#plot(u1)
#plot(u2)
#plot(u, mode = 'displacement')
#interactive()
point = (0.5,0.5)
print u.vector().array()
print u(point)
