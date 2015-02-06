
import numpy as np
from numpy import array,arange,mgrid
from numpy.linalg import det
import scipy.stats as sps

def get_lims(xx,yy):
	xm = xx.mean()
	ym = yy.mean()
	xmax = (xx.max() - xm) * 1.1 + xm
	xmin = (xx.min() - xm) * 1.1 + xm
	ymax = (yy.max() - ym) * 1.1 + ym
	ymin = (yy.min() - ym) * 1.1 + ym
	return xmax,xmin,ymax,ymin

def get_line(x1,x2,y1,y2):
	slope = (y2-y1)/(x2-x1)
	intercept = y1 - slope*x1
	def l(x):
		return slope * x + intercept
	return l

#def area_tri(xx,yy):
#	A = abs(  xx[0]*yy[1]+xx[1]*yy[2]+xx[2]*yy[0]
#			- xx[0]*yy[2]-xx[1]*yy[0]-xx[2]*yy[1] ) / 2
#	return A

def area_tri(xx,yy):
	A = abs( det(array([xx,yy,np.ones(3)])) ) / 2
	return A


def area_tri_gridest(xx,yy,grid_factor=100,print_data=False):
	ll = [get_line(xx[i],xx[j],yy[i],yy[j]) for i in range(2) for j in range(i+1,3)]
	bb_mid = [l(xx.mean())>yy.mean() for l in ll]
	
	gxmax = int(max(xx)*grid_factor+1)
	gxmin = int(min(xx)*grid_factor-1)
	gymax = int(max(yy)*grid_factor+1)
	gymin = int(min(yy)*grid_factor-1)
	
	xx_grid,yy_grid = mgrid[gxmin:gxmax+1,gymin:gymax+1]
	
	bb_grid = array([( (  ll[i](xx_grid/grid_factor)
						> yy_grid/grid_factor        )
						== bb_mid[i]) for i in range(3)]
								).prod(axis=0)
	
	A_grid = sum(bb_grid) / grid_factor**2

	if print_data:
		return A_grid, ll, xx_grid, yy_grid, grid_factor, bb_grid
	else:
		return A_grid

for i in range(10):
	xx,yy = randn(6).reshape(2,3)
	A = area_tri(xx,yy)
	A_grid, ll, xx_grid, yy_grid, grid_factor, bb_grid = area_tri_gridest(xx,yy,print_data=True)

	plot(concatenate([xx,xx[0:1]]),concatenate([yy,yy[0:1]]))

	xmax,xmin,ymax,ymin = get_lims(xx,yy)
	xlim(xmin,xmax)
	ylim(ymin,ymax)
	xls = linspace(xmin,xmax,10)
	yls = linspace(ymin,ymax,10)

	plot(xls,ll[0](xls),'r--',alpha=.2)
	plot(xls,ll[1](xls),'r--',alpha=.2)
	plot(xls,ll[2](xls),'r--',alpha=.2)

	samp_prop = 30
	bb_sample = (randint(samp_prop+1,size=xx_grid.shape)/samp_prop).astype(int).astype(bool)
	plot(xx_grid[bb_sample]/grid_factor,yy_grid[bb_sample]/grid_factor,'g.',alpha=.05)
	plot(xx_grid[bb_grid.astype(bool)&bb_sample]/grid_factor,
		yy_grid[bb_grid.astype(bool)&bb_sample]/grid_factor,'g.',alpha=.2)

	print([A,A_grid])
	print((A_grid-A)/A)
	input()
	cla()

x1,x2,x3 = xx
y1,y2,y3 = yy

x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2






m = 10**5
pts_norm = randn(6*m).reshape(m,2,3)
pts_unif = sps.uniform.rvs(size=(m,2,3))*4

AA_norm = array([area_tri(xx,yy) for xx,yy in pts_norm])
AA_unif = array([area_tri(xx,yy) for xx,yy in pts_unif])

hist(AA_norm,bins=arange(0,10,.1),alpha=.3,normed=True,color='g')
hist(AA_unif,bins=arange(0,10,.1),alpha=.3,normed=True,color='b')

xlim(0,7)




