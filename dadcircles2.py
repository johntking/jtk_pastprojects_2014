import matplotlib.pyplot as plt
from math import sqrt,pi,cos,sin,tan
from numpy import arcsin
from matplotlib import patches
from numpy.random import random,permutation,choice

th0 = { 2:0,
		3:1/12,
		4:1/8,
		5:1/20,
		6:1/10,
		7:1/16,
		8:121/1120,
		9:1789/20160,
		10:1577/40320,
		11:1051/24640,
		12:2283/49280}

def factorial(n):
	output = 1
	for x in range(1,n+1):
		output *= x
	return output

def get_angles(n,th0d):
	return [th0d[n]+i/n for i in range(n)]

def get_mod_angles(n,th0d):
	angles1 = []
	for k in range(2,n):
		angles1 += get_angles(k,th0d)
	angles2 = [(a*factorial(n))%factorial(n-1) for a in angles1]
	angles2.sort()
	angles2.append(angles2[0]+factorial(n-1))
	return angles2

def differences(mangles,precision=13):
	ndiffs = len(mangles) - 1
	diffs = []
	largest_diff = 0
	best_middles = []
	for i in range(ndiffs):
		diff = mangles[i+1]-mangles[i]
		diffs.append(diff)
		if abs(diff-largest_diff) < .1**precision:
			# i.e., diff == largest_diff (with imprecision)
			best_middles.append(mangles[i] + diff/2)
		elif diff > largest_diff:
			largest_diff = diff
			best_middles = [mangles[i] + diff/2]
	return diffs, largest_diff, best_middles

def find_th0(n,th0d):
	mangles = get_mod_angles(n,th0d)
	diffs, ld, bm = differences(mangles)
	ld_normed = ld / factorial(n)
	bm_normed = [x/factorial(n) for x in bm]
	return ld_normed,bm_normed

def get_all_th0_paths(N):
	th0s = {2:[{2:0}]}
	for n in range(3,N+1):
		th0s[n] = []
		for th0_path in th0s[n-1]:
			ldn,bmn = find_th0(n,th0_path)
			for bmx in bmn:
				new_path = {k:v for k,v in th0_path.items()}
				new_path[n] = bmx
				th0s[n].append(new_path)
	return th0s

def get_all_best_th0_paths(N,precision=13):
	th0s = {2:[{2:0}]}
	ldn_dict = {2:.5}
	for n in range(3,N+1):
		th0n = []
		for th0_path in th0s[n-1]:
			ldn,bmn = find_th0(n,th0_path)
			for bmx in bmn:
				new_path = {k:v for k,v in th0_path.items()}
				new_path[n] = bmx
				th0n.append((new_path,ldn))
		best_ldn = max([ldn for path,ldn in th0n])
		ldn_dict[n] = best_ldn
		th0s[n] = []
		for path,ldn in th0n:
			if abs(ldn-best_ldn) < .1**precision:
				th0s[n].append(path)
	return th0s,ldn_dict

# single result, N=6 and up
def gen_th0d(N,precision=13):
	th0s = {6:[{2:0,3:1/4,4:1/8,5:3/20,6:1/10}]}
	ldn_dict = {2:.5}
	for n in range(7,N+1):
		th0n = []
		for th0_path in th0s[n-1]:
			ldn,bmn = find_th0(n,th0_path)
			for bmx in bmn:
				new_path = {k:v for k,v in th0_path.items()}
				new_path[n] = bmx
				th0n.append((new_path,ldn))
		best_ldn = max([ldn for path,ldn in th0n])
		ldn_dict[n] = best_ldn
		th0s[n] = []
		for path,ldn in th0n:
			if abs(ldn-best_ldn) < .1**precision:
				th0s[n].append(path)
	return th0s[N][0],ldn_dict

def draw_rings_and_angles(N_rings):
	# setup
	figure(figsize=(10,10))
	ax = subplot(111)
	axis('off')
	subplots_adjust(left=0, bottom=0, right=1, top=1)
	# rings
	radii = {}
	for ring in range(1,N_rings+1):
		radius = sqrt(ring*(ring+1)/2)
		arc = patches.Arc((0,0),radius*2,radius*2)
		ax.add_patch(arc)
		radii[ring] = radius
	limits = radii[N_rings] * -1.05, radii[N_rings] * 1.05
	ax.set_xlim(*limits)
	ax.set_ylim(*limits)
	# rays
	th0d = gen_th0d(N_rings)[0]
	def draw_ray(inner_rad,outer_rad,angle):
		x1,y1 = inner_rad * cos(angle), inner_rad * sin(angle)
		x2,y2 = outer_rad * cos(angle), outer_rad * sin(angle)
		ax.plot([x1,x2],[y1,y2],'k')
	for ring1 in range(1,N_rings):
		th0 = th0d[ring1+1]
		r1 = radii[ring1]
		r2 = radii[ring1+1]
		for i in range(ring1+1):
			a = 2*pi * (i / (ring1+1) + th0)
			draw_ray(r1,r2,a)
	return radii,th0d

def drafill_random(N_rings):
	# setup
	figure(figsize=(10,10))
	ax = subplot(1,1,1,polar=True)
	axis('off')
	subplots_adjust(left=0, bottom=0, right=1, top=1)
	# rings
	radii = {}
	for ring in range(1,N_rings+1):
		radius = sqrt(ring*(ring+1)/2)
		#arc = patches.Arc((0,0),radius*2,radius*2)
		#ax.add_patch(arc)
		radii[ring] = radius
	#limits = radii[N_rings] * -1.05, radii[N_rings] * 1.05
	#ax.set_xlim(*limits)
	#ax.set_ylim(*limits)
	# rays
	th0d = gen_th0d(N_rings)[0]
	#def draw_ray(inner_rad,outer_rad,angle):
	#	x1,y1 = inner_rad * cos(angle), inner_rad * sin(angle)
	#	x2,y2 = outer_rad * cos(angle), outer_rad * sin(angle)
	#	ax.plot([x1,x2],[y1,y2],'k')
	regions = []
	for ring1 in range(1,N_rings):
		th0 = th0d[ring1+1]
		r1 = radii[ring1]
		dr = radii[ring1+1] - r1
		for i in range(ring1+1):
			a1 = 2*pi * (i / (ring1+1) + th0)
			da = 2*pi * 1 / (ring1+1)
			regions.append((r1,dr,a1,da))
			#draw_ray(r1,r2,a)
	for r1,dr,a1,da in regions:
		col = random(3)
		ax.broken_barh(
			[[a1+i*da/20,da/20] for i in range(20)],[r1,dr],color=col)
	return regions
#old:
def get_4c_solution(N_rings,radii,solutions=(1,1),sorter=lambda p,rd:random()):
	# note: solutions is (# solutions tried at end, # solutions tried at each step)
	# setup dictionaries
	th0d = gen_th0d(N_rings)[0]
	#regions = []
	rd = {}
	regions_by_ring = {}
	N_regions = 0
	for ring1 in range(1,N_rings):
		th0 = th0d[ring1+1]
		r1 = radii[ring1]
		dr = radii[ring1+1] - r1
		regions_by_ring[ring1+1] = []
		for i in range(ring1+1):
			a1 = 2*pi * (i / (ring1+1) + th0)
			da = 2*pi * 1 / (ring1+1)
			rd[N_regions] = {"param":(ring1+1,r1,dr,a1,da),
							 "border":[]}
			regions_by_ring[ring1+1].append(N_regions)
			N_regions += 1
	#rd = {i:
	#		{"param":r,"border":[],"col":None}
	#				for i,r in enumerate(regions)}
	#regions_by_ring = {ring:[] for ring in range(2,N_rings+1)}
	#for i in rd:
	#	regions_by_ring[rd[i]["param"][0]].append(i)
	# identify borders
	for ring in regions_by_ring:
		for i in regions_by_ring[ring]:
			i1 = rd[i]["param"][3]
			i2 = i1 + rd[i]["param"][4]
			for j in regions_by_ring[ring]:
				if j > i:
					j1 = rd[j]["param"][3]
					j2 = j1 + rd[j]["param"][4]
					if (abs(i2-j1)<.1**13 or abs(i1-j2+2*pi)<.1**13):
						rd[i]["border"].append(j)
						rd[j]["border"].append(i)
			if ring+1 not in regions_by_ring:
				continue
			for j in regions_by_ring[ring+1]:
				j1 = rd[j]["param"][3]
				j2 = j1 + rd[j]["param"][4]
				if ( (i1 > j1 and i1 < j2)
					or (i1 < j1 and i2 > j1)
					or (i2 > 2*pi and j2 > i2-2*pi and j1 < i2-2*pi)
					or (j2 > 2*pi and i2 > j2-2*pi and i1 < j2-2*pi) ):
					rd[i]["border"].append(j)
					rd[j]["border"].append(i)
	# find four-color solutions
	def select_path(N_paths_to_try):
		r = max([r for r,p in paths.items() if p!=[]])
		return paths[r][:N_paths_to_try],r
	def expand_path(path):
		new_paths = []
		region = len(path)
		border_inner_rr= [r for r in rd[region]['border'] if r<region]
		available_colors = {0,1,2,3}-{path[r] for r in border_inner_rr}
		for c in available_colors:
			new_paths.append(tuple(list(path)+[c]))
		return new_paths
	def sorter_1v(path):
		return sorter(path,rd)
	paths = {r:[] for r in range(5,max(rd)+1)}
	paths.update({4:[(0,1,2,0,3)]})
	complete_solutions = []
	paths[max(rd)] = []
	while len(complete_solutions) < solutions[0]:
		paths_to_try,r = select_path(solutions[1])
		for path in paths_to_try:
			new_paths = expand_path(path)
			paths[r].remove(path)
			if r+2 == N_regions:
				complete_solutions += new_paths
			elif new_paths != []:
				for new_path in new_paths:
					paths[r+1].append(new_path)
				paths[r+1].sort(key=sorter_1v,reverse=True)
	best_solution = sorted(complete_solutions,
								key=sorter_1v,reverse=True)[0]
	for r,c in enumerate(best_solution):
		rd[r]['color'] = c
	return rd

def random_sorter(path,rd):
	return random()
#old:
def drafill_4c(N_rings,colors = ((0,0,.8),(.8,0,0),(.8,.8,0),(1,1,1)),randcol = True,solutions=(1,1),sorter=random_sorter,center_color=3,text_on=False):
	#graph
	if randcol == True:
		colors = permutation(colors)
	figure(figsize=(10,10))
	ax = subplot(1,1,1,polar=True)
	axis('off')
	subplots_adjust(left=0, bottom=0, right=1, top=1)
	#geometric parameters
	radii = {}
	for ring in range(1,N_rings+1):
		radius = sqrt(ring*(ring+1)/2)
		radii[ring] = radius
	ax.set_ylim(0,radii[N_rings] * 1.05)
	#four-color selection
	rd = get_4c_solution(N_rings,radii,solutions=solutions,sorter=sorter)
	#graphing
	for r in rd:
		ring,r1,dr,a1,da = rd[r]["param"]
		col = colors[rd[r]["color"]]
		ax.broken_barh(
			[[a1+i*da/20,da/20] for i in range(20)],[r1,dr],color=col)
		if text_on == True:
			ax.text(a1+da/2,r1+dr/2,r+2,ha='center',va='center',fontsize=9)
	ax.broken_barh( [[i*pi/18,pi/18] for i in range(36)],
								[0,1],color=colors[center_color])
	if text_on == True:
		ax.text(0,0,'1',ha='center',va='center',fontsize=9)

#new:
def draw_4c(rd,colors=((0,0,.8),(.8,0,0),(.8,.8,0),(1,1,1)),randcol=True,text_on=False):
	#graph
	if randcol == True:
		colors = [colors[x] for x in permutation([0,1,2,3])]
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(1,1,1,polar=True)
	plt.axis('off')
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
	#geometric parameters
	max_radius = max([rd[i]['param'][1]+rd[i]['param'][2] for i in rd])
	ax.set_ylim(0,max_radius * 1.05)
	#graphing
	for i in rd:
		ring,r1,dr,a1,da = rd[i]["param"]
		col = colors[rd[i]["color"]]
		ax.broken_barh(
			[[a1+x*da/20,da/20] for x in range(20)],[r1,dr],color=col)
		if text_on == True:
			ax.text(a1+da/2,r1+dr/2,i,ha='center',va='center',fontsize=9)
	#ax.broken_barh( [[i*pi/18,pi/18] for i in range(36)],
	#							[0,1],color=colors[center_color])
	#if text_on == True:
	#	ax.text(0,0,'1',ha='center',va='center',fontsize=9)

def get_4c2(N_rings,
			solutions=(1,1),
			colpref={},
			pathorder=[],
			randomtype="ringout",
			paint=False,
			colors=((0,0,.8),(.8,0,0),(.8,.8,0),(1,1,1)),
			randcol=True,
			text_on=False,
			sorter=lambda p,rd:random()  ):
	# note: solutions is (# solutions tried at end, # solutions tried at each step)
	# setup dictionaries
	radii = {}
	for ring in range(1,N_rings+1):
		radius = sqrt(ring*(ring+1)/2)
		radii[ring] = radius
	th0d = gen_th0d(max(N_rings,6))[0]
	rd = {1:{"param":(1,0,1,pi/2,pi*2),'border':[]}}
	regions_by_ring = {1:[1]}
	counter = 1
	for inner_ring in range(1,N_rings):
		outer_ring = inner_ring + 1
		th0 = th0d[outer_ring]
		r1 = radii[inner_ring]
		dr = radii[outer_ring] - r1
		regions_by_ring[outer_ring] = []
		for i in range(outer_ring):
			counter += 1
			a1 = 2*pi * (i / (outer_ring) + th0)
			da = 2*pi * 1 / (outer_ring)
			rd[counter] = {"param":(outer_ring,r1,dr,a1,da),
							 "border":[]}
			regions_by_ring[outer_ring].append(counter)
	# borders:
	for ring in range(2,N_rings+1):
		for i in regions_by_ring[ring]:
			i1 = rd[i]["param"][3]
			i2 = i1 + rd[i]["param"][4]
			for j in regions_by_ring[ring]:
				if j > i:
					j1 = rd[j]["param"][3]
					j2 = j1 + rd[j]["param"][4]
					if (abs(i2-j1)<.1**13 or abs(i1-j2+2*pi)<.1**13):
						rd[i]["border"].append(j)
						rd[j]["border"].append(i)
			for j in regions_by_ring[ring-1]:
				j1 = rd[j]["param"][3]
				j2 = j1 + rd[j]["param"][4]
				if ( (i1 > j1 and i1 < j2)
					or (i1 < j1 and i2 > j1)
					or (i2 > 2*pi and j2 > i2-2*pi and j1 < i2-2*pi)
					or (j2 > 2*pi and i2 > j2-2*pi and i1 < j2-2*pi) ):
					rd[i]["border"].append(j)
					rd[j]["border"].append(i)
	# check colpref
	regions_by_color = {c:[] for c in range(4)}
	colpref = {i:c for i,c in colpref.items() if i in rd and c in range(4)}
	for i,c in colpref.items():
		regions_by_color[c].append(i)
	for c in range(4):
		for i in regions_by_color[c]:
			if set(rd[i]['border']) & set(regions_by_color[c]) != set():
				print("error in colpref. bordering regions have same color:",i)
				return None
	###
	predetermined_path = [i for i in pathorder if i in rd.keys() and i not in colpref]
	if len(rd) == len(predetermined_path) + len(colpref):
		random_path = []
	elif randomtype == "maxborder":
		random_path = []
		bordercount = {}
		J = set(predetermined_path) | set(colpref.keys())
		I = set(rd.keys()) - J
		for i in I:
			bordercount[i] = len(set(rd[i]['border']) & J)
		for count in I:
			maxborder = max(bordercount.values())
			i = choice([i for i,ct in bordercount.items() if ct==maxborder])
			random_path.append(i)
			del bordercount[i]
			for i2 in rd[i]['border']:
				if i2 in bordercount:
					bordercount[i2] += 1
	elif randomtype == "ringout":
		unfiltered_random_path = []
		rings = list(regions_by_ring.keys())
		for ring in sorted(rings):
			unfiltered_random_path += list(permutation(regions_by_ring[ring]))
		random_path = [i for i in unfiltered_random_path 
										if i not in predetermined_path 
										and i not in colpref]
	elif randomtype == "ringin":
		unfiltered_random_path = []
		rings = list(regions_by_ring.keys())
		for ring in sorted(rings,reverse=True):
			unfiltered_random_path += list(permutation(regions_by_ring[ring]))
		random_path = [i for i in unfiltered_random_path 
										if i not in predetermined_path 
										and i not in colpref]
	elif randomtype == "byring":
		unfiltered_random_path = []
		rings = list(regions_by_ring.keys())
		for ring in permutation(rings):
			unfiltered_random_path += list(permutation(regions_by_ring[ring]))
		random_path = [i for i in unfiltered_random_path 
										if i not in predetermined_path 
										and i not in colpref]
	else:
		random_path = list(permutation( [i for i in rd.keys()
										if i not in predetermined_path 
										and i not in colpref] ))
	pathorder = {index:i for index,i in enumerate(predetermined_path+random_path)}
	print([pathorder[index] for index in sorted(pathorder)])
	######
	#print([pathorder[index] for index in range(len(pathorder))])
	#print(colpref)
	###
	# paths: key = index of region (look up in pathorder) NEXT to be decided in current path
	paths = {index:[] for index in pathorder.keys()}
	paths[0].append(colpref)
	complete_solutions = []
	final_index = max(pathorder.keys())
	reset = False
	path_expansion_steps = 0 # purely for diagnostics / evaluating the algorithm
	solution_cycles = 0
	#def select_paths(N_paths_to_try,reset):
	#	if reset == True:
	#		index0 = min([index for index,pathlist in paths.items() if pathlist!=[]])
	#	else:
	#		index0 = max([index for index,pathlist in paths.items() if pathlist!=[]])
	#	return paths[index0][:N_paths_to_try],index0
	def expand_path(path,index0):
		new_paths = []
		region = pathorder[index0]
		border = [i for i in rd[region]['border'] if i in path]
		available_colors = {0,1,2,3}-{path[i] for i in border}
		for col in available_colors:
			new_path = {i:c for i,c in path.items()}
			new_path[region] = col
			new_paths.append(new_path)
		return new_paths
	def sorter_1v(path):
		return sorter(path,rd)
	#while len(complete_solutions) < solutions[0]:
	while solution_cycles < solutions[0]:
		if reset == False:
			index0 = max([index for index,pathlist in paths.items() if pathlist!=[]])
		else:
			reset = False
			index0 = min([index for index,pathlist in paths.items() if pathlist!=[]])
		paths_to_expand = paths[index0][:solutions[1]]
		for path in paths_to_expand:
			new_paths = expand_path(path,index0)
			paths[index0].remove(path)
			if new_paths != []:
				if index0 == final_index:
					complete_solutions += new_paths
					solution_cycles += 1
				else:
					for new_path in new_paths:
						paths[index0+1].append(new_path)
					paths[index0+1].sort(key=sorter_1v,reverse=True)
			path_expansion_steps += 1
			#print(path_expansion_steps, path)
			#print(paths)
			#print("* * *")
		if list(paths.values()).count([]) > final_index:
			return None
	best_solution = sorted(complete_solutions,
								key=sorter_1v,reverse=True)[0]
	for i,c in best_solution.items():
		rd[i]['color'] = c
	#print(path_expansion_steps)
	if paint == True:
		draw_4c(rd,colors=colors,randcol=randcol,text_on=text_on)
	return rd


def sort_random(path,rd):
	return random()

def sort_sum(path,rd):
	return sum(list(path.values()))

def sort_sumsq(path,rd):
	return sum([x**2 for x in path.values()])

def sort_sumsq_neg(path,rd):
	return -1*sum([x**2 for x in path.values()])

def sort_c0(path,rd):
	return list(path.values()).count(0)

def sort_c0n(path,rd):
	return -1*list(path.values()).count(0)

def sort_c1(path,rd):
	return list(path.values()).count(1)

def sort_c23(path,rd):
	colors = list(path.values())
	return colors.count(2) + colors.count(3)

def sort_by_yaxis_cosin(path,rd):
	score = 0 # doesn't really work...
	for region,c in path.items():
		ring,r1,dr,a1,da = rd[region]['param']
		y_value = (r1+dr/2) * sin(a1+da/2)
		region_val = cos(y_value*pi/2/5.74)
		if c in (2,3):
			region_val *= -1
		score += region_val
	return score

def sort_c0c1_repulsion(path,rd):
	score = 0
	for region,c in path.items():
		if c == 0:
			for j in rd[region]['border']:
				if j in path:
					if path[j] == 1:
						score -= 1
	return score

def sort_c0c1_attraction(path,rd):
	score = 0
	for region,c in path.items():
		if c == 0:
			for j in rd[region]['border']:
				if j in path:
					if path[j] == 1:
						score += 1
	return score

def sort_01_diag(path,rd):
	score = 0
	for region,c in path.items():
		ring,r1,dr,a1,da = rd[region]['param']
		indiag = ((a1<pi*.7 and a1+da>pi*.65) or (a1<pi*1.7 and a1+da>pi*1.65))
		c_equal_0or1 = (c in (0,1))
		if indiag ^ c_equal_0or1:
			score -= 1
		else:
			score += 1
	return score

def sort_01_hex(path,rd):
	score = 0
	for region,c in path.items():
		ring,r1,dr,a1,da = rd[region]['param']
		inhex = any((   (a1<pi*.3333 and a1+da>pi*.3333),
						(a1<pi*.6667 and a1+da>pi*.6667),
						(a1<pi and a1+da>pi),
						(a1<pi*1.3333 and a1+da>pi*1.3333),
						(a1<pi*1.6667 and a1+da>pi*1.6667),
						(a1<pi and a1+da>pi) ) )
		c_equal_0or1 = (c in (0,1))
		if inhex ^ c_equal_0or1:
			score -= 1
		else:
			score += 1
	return score

def sort_01_hex_fast(path,rd):
	score = 0
	hex_ii = [1,2,3,4,5,7,8,9,11,12,13,14,16,17,18,19,20,22,23,
	24,25,26,29,30,32,33,34,38,39,41,42,44,47,48,50,52,53,56,58,
	60,62,64,68,70,72,74,76,80,82,84,86,89,93,96,98,100,103,107,
	110,112,115,117,123,125,128,131,133,139,142,144,147,150,156,
	159,162,165,168,174,178,181,184,187,193,196,200,203,206] 
	for region,c in path.items():
		if (region in hex_ii) ^ (c in (0,1)):
			score -= 1
		else:
			score += 1
	return score
# taken from:
def inhex(a1,da):
	return any((   (a1<pi*.3333 and a1+da>pi*.3333),
						(a1<pi*.6667 and a1+da>pi*.6667),
						(a1<pi and a1+da>pi),
						(a1<pi*1.3333 and a1+da>pi*1.3333),
						(a1<pi*1.6667 and a1+da>pi*1.6667),
						(a1<pi and a1+da>pi) ) )

def sort_rings(path,rd):
	score = 0
	for i in rd:
		if i not in path:
			break
		i_color = (path[i] in (0,1))
		border_in_ring = [j for j in rd[i]['border'] if rd[j]['param'][0]==rd[i]['param'][0]]
		for j in border_in_ring:
			if j not in path:
				break
			j_color = (path[j] in (2,3))
			if i_color ^ j_color:
				score += 1
			else:
				score -= 1
	return score

def sort_ring_n(path,rd):
	score = 0
	for i in rd:
		if i not in path:
			break
		i_color = (path[i] in (0,1))
		border_in_ring = [j for j in rd[i]['border'] if rd[j]['param'][0]==rd[i]['param'][0]]
		for j in border_in_ring:
			if j not in path:
				break
			j_color = (path[j] in (2,3))
			if i_color ^ j_color:
				score += 1
			else:
				score -= 1
	return score * -1

p1={5:0,
13:0,
25:0,
42:0,
62:0,
86:0,
115:0,
147:0,
184:0,
224:0,
268:0,
317:0,
370:0,
426:0,
2:0,
16:0,
29:0,
47:0,
68:0,
93:0,
123:0,
157:0,
194:0,
235:0,
281:0,
330:0,
383:0,
440:0,
9:1,
19:1,
33:1,
52:1,
74:1,
100:1,
131:1,
165:1,
203:1,
245:1,
292:1,
342:1,
397:1,
456:1,
22:1,
38:1,
57:1,
80:1,
107:1,
139:1,
175:1,
214:1,
257:1,
305:1,
357:1,
412:1}

rd=get_4c2(30,
solutions=(10,10),
colpref={randint(465)+1:0},
pathorder=[],
randomtype="maxborder",
paint=True,
colors=("maroon","red","blue","navy"),
randcol=False,
text_on=False,
sorter=sort_c23  )

#
#get_4c2(N_rings,
#			solutions=(1,1),
#			colpref={},
#			pathorder=[],
#			randomtype="ringout",
#			paint=True,
#			colors=("green","lightgreen","blue","navy"),
#			randcol=False,
#			text_on=False,
#			sorter=sorter_6  )

#
#
#
#
#
########OLD########
##drafill_4c(20,sorter=sorter_1,solutions=100)

#drafill_4c(20,sorter=sorter_6,solutions=100,
#		colors= permutation([(0,0,.8),(.8,0,0),(.8,.8,0),(1,1,1)]))
#
#drafill_4c(20,sorter=sorter_6,solutions=100,
#		colors= permutation(['maroon','navy','lightgreen','gold']))
#
#drafill_4c(20,sorter=sorter_1,solutions=100,
#		colors= permutation([(0,0,.8),(.3,0,.5),(0,.5,.5),(.5,.5,1)]))
#
#drafill_4c(20,sorter=sorter_2,solutions=100,
#		colors= permutation([(.5,0,0),(.8,.8,0),(.7,.1,.3),(1,.5,.5)]))
#
#drafill_4c(20,sorter=sorter_2,solutions=100,
#		colors= permutation([(.6,0,0),(.8,.4,0),(.5,0,.4),(1,.8,.5)]))
#
#drafill_4c(20,sorter=sorter_2,solutions=100,
#		colors= permutation([(0,.8,0),(.2,.6,.2),(.5,.7,0),(.2,1,1)]))
#
#drafill_4c(20,sorter=random_sorter,solutions=1,
#		colors= permutation([(0,.8,0),(.8,0,0),(0,0,.8),(.7,.7,0)]))
#
#drafill_4c(20,sorter=sorter_2,solutions=1,
#		colors= permutation([(.6,0,0),(.1,.1,.1),(.6,.6,.6),(.9,.9,.9)]))
#

#drafill_4c(100,sorter=random_sorter,solutions=1,
#		colors= permutation([(.2,.2,.2),(.7,0,0),(.2,.1,.5),(.5,.8,0)]))


#Y = [0]
#for r in rd:
#	ring,r1,dr,a1,da = rd[r]['param']
#	y_value = (r1+dr/2) * sin(a1+da/2)
#	Y.append(y_value)


#for r in range(5,35):#max(rd)+1):
#	paths[r] = []
#	border_inner_ii = [i for i in rd[r]['border'] if i<r]
#	for path0 in paths[r-1]:
#		available_colors = {0,1,2,3}-{path0[i] for i in border_inner_ii}
#		for c in available_colors:
#			paths[r].append(tuple(list(path0)+[c]))
#		if available_colors == set():
#			killed_paths[r] += 1
#
### previous path generator ###
#paths = {4:[(0,1,2,0,3)]}
#killed_paths = {newi:0 for newi in range(5,35)}
#for newi in range(5,35):#max(rd)+1):
#	paths[newi] = []
#	border_inner_ii = [i for i in rd[newi]['border'] if i<newi]
#	for path0 in paths[newi-1]:
#		available_colors = {0,1,2,3}-{path0[i] for i in border_inner_ii}
#		for c in available_colors:
#			paths[newi].append(tuple(list(path0)+[c]))
#		if available_colors == set():
#			killed_paths[newi] += 1
















