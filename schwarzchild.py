import sympy as sp

sp.var('t, r, theta, phi')

xi = sp.Matrix([ [t],[r],[theta],[phi] ])

D = len(xi)

f = sp.Function('f')(r)
g = sp.Matrix([
[-f, 0, 0, 0],
[0, 1/f, 0, 0],
[0, 0, r**2, 0],
[0, 0, 0, r**2*sp.sin(theta)**2] ])

ginv = g.inv()

def Gamma(k,i,j):
    gm = 0
    for n in range(D):
        gm = gm + sp.Rational(1,2)*ginv[k,n]*(g[n,i].diff(xi[j])\
                +g[n,j].diff(xi[i])-g[i,j].diff(xi[n]))
    return gm

def R(i,j,k,l):
    rm = Gamma(i,l,j).diff(xi[k]) - Gamma(i,k,j).diff(xi[l]) 
    for n in range(D):
        rm = rm + Gamma(n,l,j)*Gamma(i,k,n) - Gamma(n,k,j)*Gamma(i,l,n)
    return rm

def Ricc(j,k):
    ric = 0
    for n in range(D):
        ric = ric + R(n,j,n,k)
    return ric

RiccMat = sp.Matrix([
[Ricc(0,0), Ricc(0,1), Ricc(0,2), Ricc(0,3)],
[Ricc(1,0), Ricc(1,1), Ricc(1,2), Ricc(1,3)],
[Ricc(2,0), Ricc(2,1), Ricc(2,2), Ricc(2,3)],
[Ricc(3,0), Ricc(3,1), Ricc(3,2), Ricc(3,3)] ])
sp.simplify(RiccMat)

R = sp.simplify((ginv*RiccMat).trace())

GMat = sp.simplify(RiccMat - sp.Rational(1,2)*g*R)

print("Metric tensor g:")
sp.pprint(g)
print("Dimension of the manifold:", D)