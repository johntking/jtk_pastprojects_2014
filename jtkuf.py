

import scipy.stats as ss
from numpy import array,arange,cumsum,mgrid
from random import choice

#def look(main_obj,n=5):
#	for i in range(n):
#		path = []; obj = main_obj
#		while True:
#			if type(obj) in (dict,collections.defaultdict):
#				key = choice(list(obj))
#				path.append(str(key))
#				obj = obj[key]
#			elif type(obj) in (list,set,tuple):
#				path.append(str(type(obj)).split("'")[1])
#				obj = choice(list(obj))
#			else:
#				print(type(obj))
#				path.append(str(obj))
#				break
#		print(' -> '.join(path[:-1]))

# reverse dictionary generation
def gen_rd1(d):
	# dictionary must have hashable values
	rd = {}
	for k,v in d.items():
		if v not in rd:
			rd[v] = {k}
		else:
			rd[v].add(k)
	return rd

def gen_rd2(d):
	# dictionary must have iterable values that contain hashable objects
	rd = {}
	for k,vv in d.items():
		for v in vv:
			if v not in rd:
				rd[v] = {k}
			else:
				rd[v].add(k)
	return rd

def look(main_obj,n=5):
	for i in range(n):
		path = []; obj = main_obj
		while True:
			if type(obj)==str:
				path.append(str(obj))
				#print(1,path)
				break
			try:
				key = choice(list(obj.keys()))
				path.append(str(key))
				obj = obj[key]
				#print(2,path)
			except AttributeError:
				try:
					type_obj = type(obj)
					obj = choice(list(obj))
					clean_type = str(type_obj).split("'")[1]
					path.append('('+clean_type+')')
					#print(3,path)
				except TypeError:
					path.append(str(obj))
					#print(4,path)
					break
				except IndexError:
					type_obj = type(obj)
					clean_type = str(type_obj).split("'")[1]
					path.append('(empty '+clean_type+')')
					break
		print(' -> '.join(path))

# directional round:
def dround(x,ndigits,direction):
	x2 = round(x,ndigits)
	if direction in (0,'-','down') and x2>x:
		return x2 - .1**ndigits
	elif direction in (1,'+','up') and x2<x:
		return x2 + .1**ndigits
	else:
		return x2


# test colors
y = 0
def t(c1,c2,c3):
	global y
	bar(0,.9,1,y-.95,color=(c1,c2,c3))
	text(.5,y-.5,(c1,c2,c3),ha='center',va='center')
	y-=1






def percint(x,T,d=0,j1=0,j2=0,spc=' '):
	if x==0:
		s1 = '0'; s2 = ''
	elif x==T:
		s1 = '100%'; s2 = '('+str(T)+')'
	else:
		if d==0:
			s1 = str(int(round(x/T*100,d)))+'%'
		else:
			s1 = str(round(x/T*100,d))+'%'
		s2 = '('+str(x)+')'
	s = ''
	if spc==' ':
		if j1==0:
			s += s1
		else:
			s += s1.rjust(j1)
		if j2==0:
			s += ' ' + s2
		else:
			s += s2.rjust(j2)
	else:
		s = s1+spc+s2
	return s


















