import numpy as np
from numpy import arange,array,ceil,linspace,cumsum,mgrid
import numpy.linalg
import scipy.stats
from collections import defaultdict

def quantile(xx,qq,null_output=0,len1output_to_float=True):
	if type(qq) in (float,int):
		qq = [qq]
	if len(xx)==0:
		output = [null_output] * len(qq)
	elif len(set(xx))==1:
		output = [list(xx)[0]] * len(qq)
	else:
		N1 = len(xx) - 1
		xx_srtd = sorted(xx)
		x_to_ii = defaultdict(list)
		for i,x in enumerate(xx_srtd):
			x_to_ii[x].append(i)
		ixd = {0:min(xx),N1:max(xx)}
		for x,ii in x_to_ii.items():
			ixd[np.mean(ii)] = x
		xx_unique = sorted(x_to_ii)
		for x1,x2 in zip(xx_unique[:-1],xx_unique[1:]):
			ii1 = x_to_ii[x1]
			ii2 = x_to_ii[x2]
			i_mid = (max(ii1)+min(ii2)) / 2
			x_mid = (x1+x2) / 2
			ixd[i_mid] = x_mid
		ii_srtd = sorted(ixd)
		output = []
		for q in qq:
			i = q * N1
			if i in ixd:
				output.append(ixd[i])
			else:
				#print(q,i)
				i1 = [i1 for i1 in ii_srtd if i>i1][-1]
				i2 = [i2 for i2 in ii_srtd if i<i2][0]
				x1 = ixd[i1]
				x2 = ixd[i2]
				x = (x2-x1)*(i-i1)/(i2-i1) + x1
				output.append(x)
	if len1output_to_float and len(output)==1:
		output = output[0]
	return output






def generate_base_matrices(valdict,units_to_include,pred_vars,resp_var):
	uu0 = array(list(units_to_include))
	Knames = array(['(cnst)']+list(pred_vars))
	X0 = array([[1]+[valdict[u][k] for k in Knames[1:]] for u in uu0])
	Y0 = array([valdict[u][resp_var] for u in uu0])
	return uu0,Knames,X0,Y0

def generate_higher_order_matrices(Knames0,X0,highest_order=2):
	Knames1 = list(Knames0)
	newXsegments = []
	for order in range(2,highest_order+1):
		Knames1.extend([kn+'^'+str(order) for kn in Knames0[1:]])
		newXsegments.append(X[:,1:].T**order)
	X1 = concatenate([X.T]+newXsegments).T
	restrictions = {i1+order*(len(Knames0)-1):i1+(1+order)*(len(Knames0)-1)
		for i1 in range(1,len(Knames0)) for order in range(highest_order-1)}
	return X1,tuple(Knames1),restrictions

def regress(X,Y):
	return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def sumsquared(xx):
	xxmu = np.mean(xx)
	return sum((xx-xxmu)**2)

def get_model_data(X,Y):
	dfr = X.shape[0] - X.shape[1]
	K = regress(X,Y)
	Yhat = X.dot(K)
	SStot = sumsquared(Y)
	R = Y - Yhat
	SSres = sum(R**2)
	MSE = SSres/dfr
	adjRsq = 1 - SSres / SStot * X.shape[0] / dfr
	return K,dfr,Yhat,R,MSE,SSres,SStot,adjRsq

#def get_submatrices(Kii,X,Y):
#	return X[:,Kii], Y[sorted(Kii)]

def lookup_model_data(X0,Y,Kii,regrdict):
	tKii = tuple(sorted(Kii))
	X1 = X0[:,tKii]
	if tKii in regrdict:
		return X1,regrdict[tKii]
	else:
		#X1,Y1 = get_submatrices(tKii,X0,Y0)
		model_data = get_model_data(X1,Y)
		regrdict[tKii] = model_data
		return X1,model_data

def evaluate_variable(i,Kii,X,K,MSE):
	j = sorted(Kii).index(i)
	ki = K[j]
	xxi = X[:,j]
	SS_xxi = sumsquared(xxi)
	sd_ki = (MSE / SS_xxi)**.5
	t_ki = abs(ki / sd_ki)
	return t_ki

"restrictions are in the form {i1:i2}, where i2 can only be included if i1 is also"

def BR_step(X0,Y,Kii1,allKii,regrdict,p_accept,p_reject,Knames,restrictions,print_res):
	dfr = X0.shape[0] - len(Kii1) - 1
	contBR = False
	best_i = 0; best_t = 0
	for i in allKii - Kii1 - {i2 for i1,i2 in restrictions.items() if i1 not in Kii1}:
		Kii2 = Kii1|{i}
		X1,mdata = lookup_model_data(X0,Y,Kii2,regrdict)
		(K1,dfr,Yhat1,R1,MSE1,SSres,SStot,Rsq) = mdata
		t_ki = evaluate_variable(i,Kii2,X1,K1,MSE1)
		if print_res: print('(accepting)',i,':',t_ki)
		if t_ki > best_t:
			best_t = t_ki
			best_i = i
	if best_t > scipy.stats.t.ppf(1-p_accept/2,dfr):
		Kii2 = Kii1|{best_i}
		contBR = True
		if print_res: print('ACCEPTED',best_i,best_t,scipy.stats.t.ppf(1-p_accept/2,dfr),dfr)
	else:
		Kii2 = Kii1
	X1,mdata = lookup_model_data(X0,Y,Kii2,regrdict)
	(K1,dfr,Yhat1,R1,MSE1,SSres,SStot,Rsq) = mdata
	worst_i = 0; worst_t = None
	for i in Kii1-{0}-{i1 for i1,i2 in restrictions.items() if i2 in Kii2}:
		t_ki = evaluate_variable(i,Kii2,X1,K1,MSE1)
		if print_res: print('(rejecting)',i,':',t_ki,scipy.stats.t.ppf(1-p_reject/2,dfr))
		if worst_t == None or t_ki < worst_t:
			worst_t = t_ki
			worst_i = i
	if worst_t!=None and worst_t < scipy.stats.t.ppf(1-p_reject/2,dfr):
		Kii3 = Kii2 - {worst_i}
		contBR = True
		if print_res: print('REJECTED',worst_i)
	else:
		Kii3 = Kii2
	return contBR,Kii3,regrdict

def best_regression(
	Knames,X0,Y,Kii={0},restrictions={},
	p_accept=.05,p_reject=.1,print_res=False):
	allKii = set(arange(1,len(Knames)))
	regrdict = {}; contBR = True
	while contBR:
		contBR,Kii,regrdict = BR_step(X0,Y,Kii,allKii,regrdict,
						p_accept,p_reject,Knames,restrictions,print_res)
		if print_res: print('-  -  -')
	Kchoice = [Knames[i] for i in sorted(Kii)]
	X1,mdata = lookup_model_data(X0,Y,Kii,regrdict)
	return Kchoice,tuple(sorted(Kii)),X1,mdata,regrdict

def best_Rsq(Knames,regrdict):
	Rsq,Kii = max([(mdata[7],Kii) for Kii,mdata in regrdict.items()])
	return Rsq,[Knames[i] for i in sorted(Kii)],Kii


# X: matrix of values for predictor variables
# Y: matrix of response values


# K,dfr,Yhat,R,MSE,SSres,SStot,Rsq = mdata





def sstd(xx0):
	if len(xx0) < 2:
		return 0
	xx = array(xx0)
	mu = np.mean(xx)
	return (sum((xx-mu)**2)/(len(xx0)-1))**.5








# directional round:
def dround(x,ndigits,direction):
	x2 = round(x,ndigits)
	if direction in (0,'-','down') and x2>x:
		return x2 - .1**ndigits
	elif direction in (1,'+','up') and x2<x:
		return x2 + .1**ndigits
	else:
		return x2


def binci(x,n,alpha=.05,prec=3):
	d=.1**prec
	pp = arange(d,1,d)
	bb0 = scipy.stats.binom.pmf(x,n,pp)
	bb1 = bb0 / np.sum(bb0)
	cs = cumsum(bb1)
	# one-sided intervals:
	up_ivl_cut = pp[cs>alpha][0] - d
	lo_ivl_cut = pp[cs<1-alpha][-1] + 2*d
	ivl_up = (up_ivl_cut,1)
	ivl_lo = (0,lo_ivl_cut)
	# interval centered by prob (p of each tail is equal):
	mid_ivl_cut0 = pp[cs>alpha/2][0] - d
	mid_ivl_cut1 = pp[cs<1-alpha/2][-1] + 2*d
	ivl_mid = (mid_ivl_cut0,mid_ivl_cut1)
	# interval centered on sample proportion:
	pbar = x/n
	if lo_ivl_cut > 2*pbar:
		ivl_cam = ivl_lo
	elif 2*pbar > 1 + up_ivl_cut:
		ivl_cam = ivl_up
	else:
		half_ivl = max(pbar-up_ivl_cut,lo_ivl_cut-pbar)
		while np.sum(bb1[(pp>pbar-half_ivl)&(pp<pbar+half_ivl)]) < 1 - alpha:
			half_ivl += d/2
		ivl_cam = (pbar-half_ivl,pbar+half_ivl)
	# shortest interval (not error-proof if prec is too small for extreme x/n):
	med = pp[cs>.5][0]
	i0 = arange(len(pp))[pp>ivl_cam[0]][0]
	j0 = arange(len(pp))[pp<ivl_cam[1]][-1]
	if bb1[i0] > bb1[j0]:
		bb_cut_rng = bb1[i0]
		j0 = arange(len(pp))[bb1>bb_cut_rng][-1]
	else:
		bb_cut_rng = bb1[j0]
		i0 = arange(len(pp))[bb1>bb_cut_rng][0]
	percentage_total = np.sum(bb1[i0:j0+1])
	while percentage_total < 1 - alpha:
		i1 = i0 - 1; b_i1 = bb1[i1]
		j1 = j0 + 1; b_j1 = bb1[j1]
		if b_i1 > b_j1:
			percentage_total += b_i1
			i0 = i1
		else:
			percentage_total += b_j1
			j0 = j1
	ivl_min = (pp[i0-1],pp[j0+1])
	# traditional normal approximation interval
	std_norm_apx = (pbar*(1-pbar)/n)**.5
	z_cut = scipy.stats.norm.ppf(1-alpha/2)
	ivl_trd = (pbar - std_norm_apx*z_cut, pbar + std_norm_apx*z_cut)
	ivls0 = (ivl_lo,ivl_up,ivl_mid,ivl_cam,ivl_min,ivl_trd)
	ivls1 = [(dround(a+d/10,prec,'-'),dround(b-d/10,prec,'+')) for a,b in ivls0]
	return ivls1,(pbar,med,std_norm_apx)

# ivls,(pbar,pmed,pstd) = binci(x,n,prec=4)

# same as above but only gives ivl_cam - much simpler!
def binci_cam(x,n,alpha=.05,prec=3):
	d=.1**prec
	pp = arange(d,1,d)
	bb0 = scipy.stats.binom.pmf(x,n,pp)
	bb1 = bb0 / np.sum(bb0)
	cs = cumsum(bb1)
	# one-sided intervals:
	up_ivl_cut = pp[cs>alpha][0] - d
	lo_ivl_cut = pp[cs<1-alpha][-1] + 2*d
	ivl_up = (up_ivl_cut,1)
	ivl_lo = (0,lo_ivl_cut)
	# interval centered by prob (p of each tail is equal):
	mid_ivl_cut0 = pp[cs>alpha/2][0] - d
	mid_ivl_cut1 = pp[cs<1-alpha/2][-1] + 2*d
	ivl_mid = (mid_ivl_cut0,mid_ivl_cut1)
	# interval centered on sample proportion:
	pbar = x/n
	if lo_ivl_cut > 2*pbar:
		ivl_cam = ivl_lo
	elif 2*pbar > 1 + up_ivl_cut:
		ivl_cam = ivl_up
	else:
		half_ivl = max(pbar-up_ivl_cut,lo_ivl_cut-pbar)
		while np.sum(bb1[(pp>pbar-half_ivl)&(pp<pbar+half_ivl)]) < 1 - alpha:
			half_ivl += d/2
		ivl_cam = (pbar-half_ivl,pbar+half_ivl)
	return ivl_cam


def comp_bin(x1,n1,x2,n2,prec=3,alpha=.05,mindiff=0):
	d=.1**prec
	pp = arange(d,1,d)
	m = len(pp)
	bb1 = pp**(x1) * (1-pp)**(n1-x1)
	bb2 = pp**(x2) * (1-pp)**(n2-x2)
	BB0 = array([[b1*b2 for b2 in bb2] for b1 in bb1])
	BB1 = BB0 / np.sum(BB0)
	xx,yy = mgrid[0:m,0:m]
	if x1/n1 == x2/n2:
		P = 1
		diff_sign = '='
	elif x1/n1 < x2/n2:
		P = np.sum(BB1) - np.sum(BB1[ xx < (yy + mindiff*m) ])
		diff_sign = '<'
	else:
		P = np.sum(BB1) - np.sum(BB1[ xx > yy - mindiff*m ])
		diff_sign = '>'
	return (x1/n1,diff_sign,x2/n2,P)

def comp_bin_plus(x1,n1,x2,n2,prec=3,alpha=.05,mindiff=0):
	p1,diff_sign,p2,P = comp_bin(x1,n1,x2,n2,prec=3,alpha=.05,mindiff=0)
	return [str(round(p1*100))+'%',str(round(p2*100))+'%',
			str(x1)+'/'+str(n1)+' vs '+str(x2)+'/'+str(n1),'p='+str(round(P,5))], P

def chisq(A,B,C,D):
	N = A + B + C + D
	AB, AC, BD, CD = A+B, A+C, B+D, C+D
	Ea = AB*AC/N
	Eb = AB*BD/N
	Ec = AC*CD/N
	Ed = BD*CD/N
	return (A-Ea)**2/Ea + (B-Eb)**2/Eb + (C-Ec)**2/Ec + (D-Ed)**2/Ed


def phi_coef(a,b,c,d):
	return (a*d - b*c) / ((a+b)*(a+c)*(b+d)*(c+d))**.5


def chisqm(M,full_out=False):
	T = M.sum()
	c_sums = M.sum(axis=0)
	r_sums = M.sum(axis=1)
	Ni,Nj = M.shape
	ii,jj = mgrid[0:Ni,0:Nj]
	expected = r_sums[ii] * c_sums[jj] / T
	Q = ((M-expected)**2 / expected).sum()
	df = (Ni-1)*(Nj-1)
	p = 1-scipy.stats.chi2.cdf(Q,df)
	if full_out==False:
		return Q,df,p
	else:
		return Q,df,p, expected, (M-expected)**2 / expected























"""
N = 100
p = .3
q = .4
ab = scipy.stats.binom.rvs(N,p,size=10000)
a = scipy.stats.binom.rvs(ab,q)
c = scipy.stats.binom.rvs((N-ab),q)
b = ab - a
d = N - (a+b+c)
chis = chisq(a,b,c,d)

cut95 = scipy.stats.chi2.ppf(.95,1)
mean(chis>cut95)

hist(chis,bins=arange(0,31,.1),normed=True,alpha=.3)
xx = arange(.1,30,.1)
plot(xx,scipy.stats.chi2.pdf(xx,1),'r-')

###

dx=.001
a,b,c,N = 63,53,20,199; d = N - (a+b+c)
pp = arange(dx,1,dx)
bb0ab = scipy.stats.binom.pmf(a+b,N,pp)
bb1ab = bb0ab / np.sum(bb0ab)
plot(pp,bb1ab*len(pp),'r-')
plot(pp,scipy.stats.norm.pdf(pp,loc=(a+b)/N,scale=sqrt((a+b)*(c+d)/N**3)),'b--')
## good enough approximation!!
# bb0ac = scipy.stats.binom.pmf(a+c,N,pp)
# bb1ac = bb0ac / np.sum(bb0ac)
# cs_ac = cumsum(bb1ac)

m = 10000
a,b,c,N = 63,53,20,199; d = N - (a+b+c)
pp = scipy.stats.norm.rvs(loc=(a+b)/N,scale=sqrt((a+b)*(c+d)/N**3),size=m)
qq = scipy.stats.norm.rvs(loc=(a+c)/N,scale=sqrt((a+c)*(b+d)/N**3),size=m)
ab = scipy.stats.binom.rvs(N,pp)
a = scipy.stats.binom.rvs(ab,qq)
c = scipy.stats.binom.rvs((N-ab),qq)
b = ab - a
d = N - (a+b+c)
chis = chisq(a,b,c,d)
cut95 = scipy.stats.chi2.ppf(.95,1)
mean(chis>cut95)


P(i is t1) = p
P(i is t2) = q
P(i is t1&t2) = p*q
P(i is t1 | i is t2) = p


"""















































# uu = array(list(set(job_exact.keys()) & TD['tsa']))
# X = array([[1]+[job_exact[u][q] for q in qq_job[:-1]] for u in uu])
# Y = array([AN_dict[FD[u]['tsa']] for u in uu])
# K = inv(X.T.dot(X)).dot(X.T).dot(Y)
# for q,k in zip(['cnst']+qq_job[:-1],K):
# 	print(q,k)



# for testing:
#Y=88,80,96,76,80,73,58,116,104,99,64,126,94,71,111,109,100,127,99,82,67,109,78,115,83
#xx1=86,62,110,101,100,78,120,105,112,120,87,133,140,84,106,109,104,150,98,120,74,96,104,94,91
#xx2=110,97,107,117,101,85,77,122,119,89,81,120,121,113,102,129,83,118,125,94,121,114,73,121,129
#xx3=100,99,103,93,95,95,80,116,106,105,90,113,96,98,109,102,100,107,108,95,91,114,93,115,97
#xx4=87,100,103,95,88,84,74,102,105,97,88,108,89,78,109,108,102,110,95,90,85,103,80,104,83
#
#Knames = ('cnst','xx1','xx2','xx3','xx4')
#X = swapaxes(array([[1]*25,xx1,xx2,xx3,xx4]),0,1)
#Y = array(Y)
#Kchoice,X1,mdata,regrdict = best_regression(Knames,X,Y)
#
#











