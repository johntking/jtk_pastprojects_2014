import matplotlib.pyplot as plt
from matplotlib import patches
from math import sqrt,pi,cos,sin,tan
from numpy import arcsin

#r1 = sqrt(1/pi)
r2 = sqrt(2/pi)
r3 = sqrt(46/7/pi)
r4 = sqrt(18/pi)
r5 = 3.5856730447824 # see below for numerical calculation
r51 = sqrt(r5**2 + 1/pi) - 1/sqrt(pi)
r52 = sqrt(r5**2 + 1/pi)
r53 = sqrt(r5**2 + 1/pi) + 1/sqrt(pi)
r6 = sqrt(100/3/pi)
r7 = sqrt((64 - 46/3)/pi)
r8 = 3.7609674207875
r9 = sqrt(64/pi)
r10 = 0.8136105555667


a11 = 2*pi * 1/32
a12 = 2*pi * (.5 - 1/32)
a13 = 2*pi * (.5 + 1/32)
a14 = 2*pi * (1 - 1/32)
a20 = 2*pi * (.5 - 1/16)/5
a21 = a11 + a20
a22 = a11 + 2*a20
a23 = a11 + 3*a20
a24 = a11 + 4*a20
a25 = a13 + a20
a26 = a13 + 2*a20
a27 = a13 + 3*a20
a28 = a13 + 4*a20
a30 = 2*pi*2.5/46
a31 = 2*pi * 3.5/46
a32 = 2*pi * (3.5+5)/46
a33 = 2*pi * (3.5+5+6)/46
a34 = 2*pi * (3.5+5+6+5)/46
a35 = 2*pi * (3.5+5+6+5+7)/46
a36 = 2*pi * (3.5+5+6+5+7+5)/46
a37 = 2*pi * (3.5+5+6+5+7+5+6)/46
a38 = 2*pi * (3.5+5+6+5+7+5+6+5)/46
a40 = arctan(1/(sqrt(pi)*r5))
a41 = 2*pi * (3.5 + 2.5) / 46
a42 = 2*pi * (3.5 + 11 + 2.5) / 46
a43 = 2*pi * (3.5 + 23 + 2.5) / 46
a44 = 2*pi * (3.5 + 34 + 2.5) / 46
a50 = 2*pi * 3/2 / 46
a51 = a32 + a50
a52 = a33 - a50
a53 = a36 + a50
a54 = a37 - a50

def intersect(r1,r2,d):
	return sqrt(r1**2*r2**2 - .25*(r1**2+r2**2-d**2)**2) / d

a61 = arcsin(intersect(r8,r10,r9) / r8)
a62 = pi - a61
a63 = pi + a61
a64 = 2*pi - a61
a70 = arcsin(intersect(r9,r10,r9) / r10)
a71 = pi - a70
a72 = pi + a70
a73 = 2*pi-a70
a74 = a70
a80 = 2 / (r8**2 - r4**2)
a81 = a31 - a80
a82 = a34 + a80
a83 = a35 - a80
a84 = a38 + a80

deg = lambda angle: angle * 180 / pi

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
xlim(-5,5)
ylim(-5,5)
axis('off')
subplots_adjust(left=0, bottom=0, right=1, top=1)

def draw_arc(center,radius,thetas=None):
	if thetas == None:
		arc = patches.Arc(center,radius*2,radius*2)
	else:
		arc = patches.Arc(center,radius*2,radius*2,
			theta1=deg(thetas[0]),theta2=deg(thetas[1]))
	ax.add_patch(arc)

#for circle_rad in (r1,r2,r4,r9):
for circle_rad in (r2,r4,r9):
	draw_arc((0,0),circle_rad)
for rad,thetas in ( (r3,(a11,a12)),
					(r3,(a13,a14))
	):
	draw_arc((0,0),rad,thetas=thetas)
for a in (a41,a42,a43,a44):
	p,q = r52*cos(a),r52*sin(a)
	draw_arc((p,q),1/sqrt(pi))
	draw_arc((0,0),r5,thetas=(a-a30,a-a40))
	draw_arc((0,0),r5,thetas=(a+a40,a+a30))
draw_arc((0,0),r7,thetas=(a32,a33))
draw_arc((0,0),r7,thetas=(a36,a37))
draw_arc((0,0),r6,thetas=(a51,a52))
draw_arc((0,0),r6,thetas=(a53,a54))
draw_arc((r9,0),r10,thetas=(a71,a72))
draw_arc((-1*r9,0),r10,thetas=(a73,a74))
for aa in ((a61,a31),(a34,a62),(a63,a35),(a38,a64)):
	draw_arc((0,0),r8,thetas=aa)


#arcs = []
#for circle_rad in (r1,r2,r4,r9):
#	arcs.append(patches.Arc((0,0),circle_rad*2,circle_rad*2))
#arcs.append(patches.Arc((0,0),r3*2,r3*2,theta1=deg(a11),theta2=deg(a12)))
#arcs.append(patches.Arc((0,0),r3*2,r3*2,theta1=deg(a13),theta2=deg(a14)))
#
#arcs.append
#
#for arc in arcs:
#	ax.add_patch(arc)

def draw_ray(inner_rad,outer_rad,angle):
	x1,y1 = inner_rad * cos(angle), inner_rad * sin(angle)
	x2,y2 = outer_rad * cos(angle), outer_rad * sin(angle)
	ax.plot([x1,x2],[y1,y2],'k')

for a in (a11,a12,a13,a14):
	draw_ray(r2,r4,a)
draw_ray(r2,r3,.5*pi)
draw_ray(r2,r3,1.5*pi)
for a in (a21,a22,a23,a24,a25,a26,a27,a28):
	draw_ray(r3,r4,a)
for a in (a31,a32,a33,a34,a35,a36,a37,a38):
	draw_ray(r4,r9,a)
for a in (a41,a42,a43,a44):
	draw_ray(r4,r51,a)
	draw_ray(r53,r9,a)
draw_ray(r7,r9,pi/2)
draw_ray(r7,r9,pi*3/2)
for a in (a51,a52,a53,a54):
	draw_ray(r4,r7,a)
draw_ray(r4,r9-r10,0)
draw_ray(r4,r9-r10,pi)
for a in (a81,a82,a83,a84):
	draw_ray(r4,r8,a)

plot([-1*r2,r2],[0,0],'k')

#draw_ray(r2,r4,a11)
#draw_ray(r2,r4,a12)
#draw_ray(r2,r4,a13)
#draw_ray(r2,r4,a14)

#draw_ray(r2,r3,.5*pi)
#draw_ray(r2,r3,1.5*pi)
#draw_ray(r3,r4,a21)
#draw_ray(r3,r4,a22)
#draw_ray(r3,r4,a23)
#draw_ray(r3,r4,a24)
#draw_ray(r3,r4,a25)
#draw_ray(r3,r4,a26)
#draw_ray(r3,r4,a27)
#draw_ray(r3,r4,a28)
#draw_ray(r4,r9,a31)
#draw_ray(r4,r9,a32)
#draw_ray(r4,r9,a33)
#draw_ray(r4,r9,a34)
#draw_ray(r4,r9,a35)
#draw_ray(r4,r9,a36)
#draw_ray(r4,r9,a37)
#draw_ray(r4,r9,a38)
#draw_ray(r4,r9,a41)#
#draw_ray(r4,r9,a42)#
#draw_ray(r4,r9,a43)#
#draw_ray(r4,r9,a44)#
#


#
#r5 = solve_at1(area_1v,1,(3,4),13)
#
## r5 = 3.5856730447824
#
#r10 = solve_at1(area2,1,(.5,2),13)
#
## r10 = 0.8136105555667
#
#r8 = solve_at1(area3_1v,3,(3.7,4.5),13)
#
## r8 = 3.7609674207875
#
#a61 = arcsin(intersect(r8,r10,r9) / r8)
#
#a70 = arcsin(intersect(r9,r10,r9) / r10)


def area3_1v(r):
	return area3(r,r10,r9) + 7*64/46 -7*pi/46 * r**2

def area3(r1,r2,d):
	xbound = sqrt(r1**2*r2**2 - .25*(r1**2+r2**2-d**2)**2) / d # = intersect(..)
	t1 = r1**2*arcsin(xbound/r1)
	t2 = r2**2*arcsin(xbound/r2)
	t3 = xbound * sqrt(r1**2-xbound**2)
	t4 = xbound * sqrt(r2**2-xbound**2)
	t5 = -2 * d * xbound
	return t1 + t2 + t3 + t4 + t5

# area3(r,r9,r9) = area2(r)

def area2(r):
	xbound_mod = sqrt(1 - r**2/4/r9**2)
	t1 = r9**2 * arcsin(xbound_mod *r/r9)
	t2 = r**2 * arcsin(xbound_mod)
	t3 = r9*r*xbound_mod
	#print(mod_limit,t1,t2,t3)
	return t1 + t2 - t3

def area_1v(r):
	return area(r,a32 - a41)

def area(r,theta):
	alpha = arctan(1/(sqrt(pi)*r))
	ip = 1/sqrt(pi)
	t1 = (theta-alpha)/2 * (r**2-r4**2)
	t2 = .5 * ip * r**3 /(r**2+ip**2)
	t3 = .5/pi * alpha - .25
	t4 = .5 * ip**3 / (r**2+ip**2) * r
	t5 = -.5 * alpha * r4**2
	return t1+t2+t3+t4+t5

def solve1(f1,val,guessivl):
	best_vars = None
	best_zero = 1
	for v in guessivl:
		new_zero = (f1(v) - val)**2
		if best_vars == None or new_zero < best_zero:
			best_zero = new_zero
			best_vars = v
	return best_vars

def solve_at1(f1,val,min_max_val,precision,printsteps=False):
	ivl_len = min_max_val[1] - min_max_val[0]
	basestep = ivl_len/10
	start_precision1 = log(basestep) / log(10) * -1
	start_precision2 = int(start_precision1)
	if start_precision1 > 0:
		start_precision2 += 1
	first_iteration = True
	if printsteps == True:
		print("args:",val,min_max_val,precision)
		print("sp1,sp2:",start_precision1,start_precision2)
	for prec in arange(start_precision2,precision+1):
		step = .1**prec
		if first_iteration == True:
			first_iteration = False
			#print('A')
			min_max_step = min_max_val[0],min_max_val[1],step
			rng = arange(min_max_val[0],min_max_val[1]+step,step)
		else:
			#print('B')
			min_max_step = (max(guess - 6 * step,min_max_val[0]),
							min(guess + 7 * step,min_max_val[1]), 
									step)
			rng = arange(*min_max_step)
		guess = solve1(f1,val,rng)
		if printsteps == True:
			print("p/mms/g:",prec,min_max_step,guess)
	return round(guess,precision)



def solve2(f1,f2,guessranges):
	if len(guessranges) != 2:
		return "error: this function only works with functions with 2 dep vars"
	best_vars = None
	best_zero = 1
	for v1 in guessranges[0]:
		for v2 in guessranges[1]:
			new_zero = (f1(v1,v2) - f2(v1,v2))**2
			if best_vars == None or new_zero < best_zero:
				best_zero = new_zero
				best_vars = (v1,v2)
	return best_vars













