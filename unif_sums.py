
def bnml(n,k):
	return prod(arange(1,n+1)/concatenate((arange(1,k+1),arange(1,n-k+1))))

figure(figsize=(10,10))

for n in range(1,11):
	M = []
	Y = []
	Xidx = [(i,j) for i in range(n) for j in range(n)]
	
	for i in range(n):
		for l in range(n):
			m = [bnml(j,l)*(i)**(j-l)*(-1)**(i2-i+1) if (i2==i or i2==i-1) and j>=l else 0 for i2,j in Xidx]
			if l==n-1:
				y = bnml(n,i)*(-1)**(i+1) / prod(arange(1,n))
			else:
				y = 0
			M.append(m)
			Y.append(y)
	
	X = inv(array(M)).dot(Y)
	coef_dict = dict(zip(Xidx,X))
	
	def get_f(coefficients):
		def f(x):
			#return sum([a*(x*n)**i for i,a in enumerate(coefficients)]) * n
			return sum([a*x**i for i,a in enumerate(coefficients)])
		return f
	
	F = [get_f([coef_dict[(i,j)] for j in range(n)]) for i in range(n)]
	
	sd = sqrt(1/(12*n))

	xx0 = linspace(-.2,n+.2,30)
	#xx0 = linspace(-.2,1.2,1000)
	for i,f in enumerate(F):
		xx1 = linspace(i,i+1,30)
		c = linspace(.05,.95,n)[i]
		#plot(xx1/n-.5,[f(x)*n for x in xx1],'-',color=(c,0,1-c),lw=3,alpha=.4)
		#plot(xx0/n-.5,[f(x)*n for x in xx0],'--',color=(c,0,1-c),lw=1,alpha=.2)
		sd = sqrt(1/(12*n))
		plot((xx1/n-.5)/sd,[f(x)*n*sd for x in xx1],'-',color=(c,0,1-c),lw=3,alpha=.4)
		plot((xx0/n-.5)/sd,[f(x)*n*sd for x in xx0],'--',color=(c,0,1-c),lw=1,alpha=.2)
	
	plot([-4,4],[0,0],'k--')

	max_height = F[n//2](n/2) * n*sd
	
	#xlim(-.2,n+.2)
	xlim(-4,4)
	ylim(-max_height/10,max_height*1.2)
	#print(max_height)
	input('p.d.f. for ∑U/n, n='+str(n))
	#cla()

xx = linspace(-4,4,50)
plot(xx,1/sqrt(2*pi)*e**(-xx**2/2),'go',alpha=.7)


## older code:



def f2(y):
	if y < 1/2:
		return (2*y)**2 / 2
	else:
		return (2*y)**2 / 2 - (2*y-1)**2

def f3(y):
	if y < 1/3:
		return (3*y)**3 / 6
	elif y < 2/3:
		return (3*y)**3 / 6 - (3*y-1)**3 / 2
	else:
		return (3*y)**3 / 6 - (3*y-1)**3 / 2 + (3*y-2)**3 / 2

def f4(y):
	if y < 1/4:
		return (4*y)**4 / 24
	elif y < 2/4:
		return (4*y)**4 / 24 - (4*y-1)**4 / 6
	elif y < 3/4:
		return (4*y)**4 / 24 - (4*y-1)**4 / 6 + (4*y-2)**4 / 4
	else:
		return (4*y)**4 / 24 - (4*y-1)**4 / 6 + (4*y-2)**4 / 4 - (4*y-3)**4 /6

def f5(y):
	if y < 1/5:
		return (5*y)**5 / 120
	elif y < 2/5:
		return (5*y)**5 / 120 - (5*y-1)**5 / 24
	elif y < 3/5:
		return (5*y)**5 / 120 - (5*y-1)**5 / 24 + (5*y-2)**5 /12
	elif y < 4/5:
		return (5*y)**5 / 120 - (5*y-1)**5 / 24 + (5*y-2)**5 /12 - (5*y-3)**5 /12
	else:
		return (5*y)**5 / 120 - (5*y-1)**5 / 24 + (5*y-2)**5 /12 - (5*y-3)**5 /12 + (5*y-4)**5 /24


def f6(y):
	if y < 1/6:
		return (6*y)**6 /720
	elif y < 2/6:
		return (6*y)**6 /720 - (6*y-1)**6 /120
	elif y < 3/6:
		return (6*y)**6 /720 - (6*y-1)**6 /120 + (6*y-2)**6 /12
	elif y < 4/6:
		return (6*y)**6 /720 - (6*y-1)**6 /120 + (6*y-2)**6 /12 - (6*y-3)**6 /12
	elif y < 5/6:
		return (6*y)**6 /720 - (6*y-1)**6 /120 + (6*y-2)**6 /12 - (6*y-3)**6 /12 + (6*y-4)**6 /24
	else:
		return (6*y)**6 /720 - (6*y-1)**6 /120 + (6*y-2)**6 /12 - (6*y-3)**6 /12 + (6*y-4)**6 /24


def bnml(n,x):
	n_nx = min(x,n-x)
	output = 1
	for i in range(n_nx):
		output *= (n-i) / (1+i)
	return output

def fac(x):
	output = 1
	for i in range(1,x+1):
		output*=i
	return output

def fn(y,n):
	p = 0
	for i in range(n):
		if y > i/n:
			p += (-1)**i* (n*y-i)**n /fac(i) /fac(n-i)
		else:
			break
		print(i,p)
	return p





S = 0
for m in range(n):
	M = 0
	for k in range(m+1):
		K = U(m/n,k,n)*(m/(n+1)+k/n/(n+1)) - U((m-1)/n,k,n)*((m-1)/(n+1)+k/n/(n+1))
		M+=K
	S += M
	print(m,M)



def mu(n):
	return sum(random.random(n))/n



def xbar(n):
	return (mu(n) - .5) * sqrt(12 * n)
	








zip((5,0.0166640167596),
(10,0.00833110176136),
(15,0.00555681774009),
(20,0.00416834219521))



S = {
5:
[0.0166413602815,
0.0167456988674,
0.0166352716034,
0.016780340971,
0.0167664457111,
0.016524695274,
0.0165201192847,
0.0167079457737,
0.0166542730693],
10:
[0.00830571578788,
0.00824733677567,
0.00835121459557,
0.00835339721805,
0.00829493782061,
0.00830978845667,
0.00830478713466,
0.00830216958066,
0.00831587079476,
0.00840842208377,
0.00833239111815,
0.00839763721116,
0.00838596230492,
0.00830615383431,
0.0083588895056,
0.00834693576432,
0.00832038436235,
0.00831783735546],
15:
[0.00556648571726,
0.0055307274995,
0.00561690208623,
0.00550931044692,
0.00553179806733,
0.00553967845885,
0.00557647023598,
0.00559118487604,
0.00555489008579,
0.00555072992702],
20:
[0.00419611832732,
0.00416515571437,
0.0041479985286,
0.00416543349386,
0.0041859837432,
0.00417111469378,
0.0041682714872,
0.00415676412085,
0.00415662744249,
0.00416995440043],
50:
[0.00167280005911,
0.00166427775815,
0.00166332278534,
0.00167341665633,
0.00166243138624,
0.00166670008441,
0.00166269451721,
0.00166858604485,
0.00168209554463,
0.00168259968754],
100:
[0.000833728221351,
0.00083329612395,
0.000833332569665,
0.000833634325208,
0.000829228197026,
0.000830949474261,
0.000829112895345,
0.000832586362497,
0.000833216508612,
0.000836239538969],
150:
[0.000549043227834,
0.00055166966741,
0.000553336176882,
0.000558440601012,
0.000555811893659,
0.000554720568038,
0.000559905728725,
0.00055057328348,
0.000556204340933,
0.000557256162462],
200:
[0.000418645217783,
0.000417731116594,
0.000416371749149,
0.000413989418012,
0.000418069025189,
0.000414687331603,
0.000418661929514,
0.000420344683526,
0.00041669579147,
0.000417831379083]
}



zip((5, 0.0166640167596),
(10, 0.00833110176136),
(15, 0.00555681774009),
(20, 0.00416834219521),
(50, 0.00166989245238),
(100, 0.000832532421688),
(150, 0.000554696165044),
(200, 0.000417302764192))



nn = (5, 10, 15, 20, 50, 100, 150, 200)
SS = (0.0166640167596, 0.00833110176136, 0.00555681774009, 0.00416834219521, 0.00166989245238, 0.000832532421688, 0.000554696165044, 0.000417302764192)















			
