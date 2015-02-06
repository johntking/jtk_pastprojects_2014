#from collections import defaultdict,Counter
#from datetime import datetime
import numpy as np
from numpy import arange,array,ceil,linspace
from numpy.random import randint
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import scipy.stats as ss

def is_it(x):
	if type(x) in (tuple,list,np.ndarray): return True
	else: return False

font36 = FontProperties()
font36.set_family('sans-serif')
font36.set_name('Helvetica')
font36.set_size(36)
font36b = font36.copy()
font36b.set_weight('bold')
font72 = font36.copy()
font72.set_size(72)
font24 = font36.copy()
font24.set_size(24)
font18 = font36.copy()
font18.set_size(18)
font14 = font36.copy()
font14.set_size(14)
font12 = font36.copy()
font12.set_size(12)
font10 = font36.copy()
font10.set_size(10)
font8 = font36.copy()
font8.set_size(8)

master_color_dict = {
	'lpr': (.7,.7,1),
	'mgr': (.3,.7,.3),
	'lyw': (.9,.9,.5),
	'tan': (.9,.8,.7),
	'gld': (.9,.7,.1),
	'lgr': (.6,.8,.6),
	'lrd': (.9,.5,.5),
	'mrd': (.8,.3,.3),
	'aqu': (.5,.8,.8),
	'mbl': (.5,.6,.9),
	'mgy': (.7,.7,.7),
	'pnk': (.9,.6,.8),
	'mpr': (.7,.5,.9),
	}

def_colors1 = {(0,i):master_color_dict[c] for i,c in enumerate([
	'lpr','lpr','lgr','lrd','gld','mbl','lyw','mpr','tan','aqu','pnk',
					])}
def_colors1[0] = master_color_dict['lpr']

"""
This script has utility graphing functions that take data in the following structure:

[	Group Name (= page header),
	[
		[	Graph/Question Title,
			Value Axis Tick Info *,
			[	[Tick Label, Value], ... ]
		],
		[ ... ], ...
	]
]

* for now, tick info will simply be ["percentage",x axis label,count_total], 
  which will trigger an automatic tick generation.
"""



"""
NOTES:
  - The "inputs" (barcolor, graph_params, ...) can take single values (as in the
	default) or they can take dictionaries.  Make sure to include a default value
	with a key of 0.  Other keys should correspond to the q_idx (simply the
	order in which the graph/subplot appears in the graph_data0). 
  - The letter "q" is often used to represent a single subplot, or the information
    that pertains to a single subplot.  This comes from the origins of this code
    in visualizing survey results, where we have questions (q/qq) and answers (a/aa).
    I'm pretty sure I did not, however, use 'a' for the answers (instead I use x/y
    or lab/label).  
  - The keyword "text_space_override" gives the option of setting bar_width and 
    y_interbar_buffer automatically by the number of lines in the labels.  
  - In the measurement system used in this script, Helevetica pt12 has a height of
    .479 and a space height of .234.  (just FYI...)
"""


char_width12a = { # helvetica size 12
	"'":                        0.13036,   #0.03259,
	'ijl':                      0.15152,   #0.03788,
	' ftI./!,:;':               0.18968,   #0.04742,
	'r-()':                     0.2274,   #0.05685,
	'"':                        0.24224,   #0.06056,
	'*':                        0.26556,   #0.06639,
	'^':                        0.27188,   #0.06797,
	'cksvxyzJ':                 0.34096,   #0.08524,
	'abdeghnopquL0123456789#$': 0.3794,   #0.09485,
	'=':                        0.39836,   #0.09959,
	'FTZ':                      0.41684,   #0.10421,
	'ABEKPSVXY&':               0.45528,   #0.11382,
	'wCDHNRU':                  0.49272,   #0.12318,
	'GOQ':                      0.53092,   #0.13273,
	'mM':                       0.56836,   #0.14209,
	'%':                        0.60728,   #0.15182,
	'W':                        0.64448,   #0.16112,
	'@':                        0.6914, } #0.17285,   }
char_width12b = {c:x/4 for cc,x in char_width12a.items() for c in cc}
char_width12b.update({'default':0.1,'char_hgt':.120,'spc_hgt':.0587})
char_width14b = {c:x*14/12 for c,x in char_width12b.items()}
char_width14b.update({'default':0.117,'char_hgt':.140,'spc_hgt':.075})
char_width18b = {c:x*18/12 for c,x in char_width12b.items()}
char_width18b.update({'default':0.5,'char_hgt':.180,'spc_hgt':.0964})
# untested:
char_width8b = {c:x*8/12 for c,x in char_width12b.items()}
char_width8b.update({'default':0.07,'char_hgt':.080,'spc_hgt':.04})



def measure_string(s,cw_dict=char_width12b):
	return sum( cw_dict[c] if c in cw_dict else cw_dict['default'] for c in s )

def measure_string_hgt(s,cw_dict=char_width12b):
	Nbreaks = s.count('\n')
	return cw_dict['char_hgt'] * (Nbreaks+1) + cw_dict['spc_hgt'] * Nbreaks

def multiply_by_pt7(lab_spacing):
	return lab_spacing * .7

def multiply_by(p):
	def multiply_by_p(lab_spacing):
		return lab_spacing * p
	return multiply_by_p

def determine_y_space(
	ylabels,
	Nsubbars,
	x_axis_info,
	cw_dict = char_width12b,
	extra_label_space = 'auto',
	bar_width_function = multiply_by(.7),
	y_edge_buffer = .1  ):
	# customizable kwargs:
	# - extra_label_space: space between labels.  2 ~ double spacing.
	# - bar_width_function: when labels get further apart, the expansion
	#   is shared in the graph between the bars and the space between the bars.
	#   this function determines how much of this expansion is given to the 
	#   bars.  e.g., if it returns 1, then there is no interbar buffer.  
	# max_lab_len is for the subplot specs: left_margin = .03 + mll/x_fig
	max_lab_len = max([ measure_string(sublab,cw_dict=cw_dict)
						for lab in ylabels
						for sublab in str(lab).split('\n')  ])
	# the rest is for the y values of the subplot and its overall height:
	lb_cts = [lab.count('\n') if type(lab)==str else 0 for lab in ylabels]
	lb_tups = {(l1,l2) for l1,l2 in zip(lb_cts[:-1],lb_cts[1:])}
	max_av_tup = max(l1+l2 for l1,l2 in lb_tups) / 2
	if extra_label_space=='auto' and (Nsubbars==1 or 'iqr' in x_axis_info
											or 'iqr alpha' in x_axis_info
											or 'stacked' in x_axis_info):
		els = 2.5
	elif extra_label_space=='auto':
		els = Nsubbars * 1.2 + 1.5
	else:
		els = extra_label_space
	lab_spacing = (max_av_tup * cw_dict['spc_hgt']        # label linebreaks
				+ (max_av_tup+1) * cw_dict['char_hgt']    # label line (chars)
				+ els * cw_dict['spc_hgt']) # space btwn labels
	bar_width =  bar_width_function(lab_spacing)
	interbar_buffer =  lab_spacing - bar_width
	ymin = 0
	N_yy = len(ylabels)
	ymax = 2*y_edge_buffer + N_yy*bar_width + (N_yy-1)*interbar_buffer
	y_ticks = y_edge_buffer + bar_width/2 + arange(N_yy) * lab_spacing
	# with the following the subplot's bars and ylabels can be drawn:
	return max_lab_len,ymin,ymax,y_ticks,bar_width

def make_GID(
	graph_data0,
	pg_len_limit,
	gpi={},
	extra_label_space='auto',
	subplot_title_pt=14,
	figure_title_pt=18,
	print_graph_numbers=True,
	bar_width_function=multiply_by(.7),
	print_continued=True,      ):
	#print(graph_data0)
	GID = { q_idx+1: 
		{   'q_name':q_data[0],
			'x_axis_info':q_data[1],
			'xx':[x for ylab,x in q_data[2]],
			'ylabels':[ylab for ylab,x in q_data[2]],
			'group_idx':group_idx,
			'N_yy':len(q_data[-1])   } # = len(q_data[2])
				for q_idx,(q_data,group_idx) in enumerate(
				[  (q_data,group_idx) 
				for group_idx,(group_name,group_data) in enumerate(graph_data0)
				for q_data in group_data] ) }
	for q_idx,graph_info in GID.items():
		#if set(len(x) for x in graph_info['xx']) == 1:
		#	graph_info['xx'] = [x[0] for x in graph_info['xx']]
		ylabels = graph_info['ylabels']
		graph_info['Nsubbars'] = (len(graph_info['xx'][0])
							if (is_it(graph_info['xx'][0]))
							else 1 )
		#print(q_idx,graph_info['Nsubbars'])
		max_lab_len,ymin,ymax,y_ticks,bar_width = determine_y_space(ylabels,
									graph_info['Nsubbars'],
									graph_info['x_axis_info'],
									bar_width_function=bar_width_function,
									extra_label_space=extra_label_space)
		graph_info['subplot_hgt'] = ymax-ymin
		graph_info['max_lab_len'] = max_lab_len
		graph_info['y_axis_info'] = ymin,ymax,y_ticks,bar_width
		# the following assumes 14pt helvetica for subplot title and is APPROXIMATE:
		if graph_info['q_name']=='':
			if print_graph_numbers:
				subplot_title_nlines = 1
			else:
				subplot_title_nlines = 0
		else:
			subplot_title_nlines = graph_info['q_name'].count('\n') + 1
		graph_info['super_buffer'] = .2 + subplot_title_nlines * (
						char_width14b['char_hgt'] + char_width14b['spc_hgt'] )
		# assuming single-spaced, 12pt helv x-axis label and tick labels:
		graph_info['sub_buffer'] = .5 + char_width12b['char_hgt']*2
		# assuming 18pt helv figure title:
		header_title = graph_data0[graph_info['group_idx']][0]
		if header_title == '':
			ftitle_nlines = 0
		else:	
			ftitle_nlines = header_title.count('\n') + 1
		graph_info['header_buffer'] = .2 + ftitle_nlines * (
						char_width18b['char_hgt'] + char_width18b['spc_hgt'] )
	########################################
	# gpi allows for manual override of GID parameters for any or all q_idx
	for q_idx,graph_info_supp in gpi.items():
		if q_idx in GID:
			for param in graph_info_supp:
				GID[q_idx][param] = graph_info_supp[param]
	if 0 in gpi:
		for q_idx in GID:
			for param in gpi[0]:
				GID[q_idx][param] = gpi[0][param]
	########################################
	# here we distribute our graphs to pages
	graph_page_data = {} # {page_idx: [figure_title,[q_indices],figure_height]}
	prev_group_idx = -1 # dummy -1, group_idx starts at 0
	q_idx = 1; page_idx = 0 # note that q_idx and page_idx BOTH start at 1
	while q_idx in GID:
		group_idx = GID[q_idx]['group_idx']
		add_hgt = sum(GID[q_idx][param] for param in 
							('subplot_hgt','super_buffer','sub_buffer'))
		if group_idx != prev_group_idx: # new page b/c new group:
			page_idx += 1
			group_name = graph_data0[group_idx][0]
			graph_page_data[page_idx] = [group_name,[q_idx],
										add_hgt + GID[q_idx]['header_buffer'] ]
		elif (pg_len_limit==None or # same page:
				graph_page_data[page_idx][2] + add_hgt <= pg_len_limit):
			graph_page_data[page_idx][1].append(q_idx)
			graph_page_data[page_idx][2] += add_hgt
		else: # new page b/c pg_len_limit is exceeded:
			page_idx += 1
			if group_name!=None and print_continued:
				group_name = group_name + ' (continued)'
			graph_page_data[page_idx] = [group_name,[q_idx],
										add_hgt + GID[q_idx]['header_buffer'] ]
		prev_group_idx = group_idx
		q_idx += 1
	return GID,graph_page_data

def make_custom_figure(fz,xy_for_each_ax):
	# xy_for_each_ax starts with the TOP ax since that's
	# what we typically get to first when processing the
	# data, but the coordinates go from the BOTTOM left 
	# to be consisten with matplotlib.  
	fig = plt.figure(figsize=fz)
	x_fig,y_fig = fz
	axx = []
	for i,(x0,y0,x1,y1) in enumerate(xy_for_each_ax):
		gs = gridspec.GridSpec(1, 1,
				bottom = y0/y_fig,
				top =    y1/y_fig,
				left =   x0/x_fig,
				right =  x1/x_fig    )[0]
		ax = fig.add_subplot(gs)
		axx.append(ax)
	return fig,axx

def make_background_bars(ax,xmin,xmax,xstep,x_ticks,ymin,ymax):
	bckgrnd_bar_midgap = xmax/100
	bckgrnd_bar_edgegap = 0
	bb_left_edges = x_ticks[:-1] + bckgrnd_bar_midgap/2
	bb_widths = [xstep - bckgrnd_bar_midgap] * len(bb_left_edges)
	bb_left_edges[0] = xmin + bckgrnd_bar_edgegap
	bb_widths[0] = bb_left_edges[1] - bb_left_edges[0] - bckgrnd_bar_midgap
	bb_widths[-1] = xmax - bb_left_edges[-1] - bckgrnd_bar_edgegap
	ax.bar( bottom=ymin,  height=[ymax-ymin]*len(x_ticks[1:]),
			left=bb_left_edges, width=bb_widths, 
			color=(0,0,0),alpha=.03,edgecolor=(1,1,1))

def get_xstep(m):
	m_order = np.floor(np.log(m)/np.log(10))
	unit_m = m/10**m_order
	if unit_m > 4:
		xstep = 1 * 10**m_order
	elif unit_m > 1.5:
		xstep = .5 * 10**m_order
	else:
		xstep = .2 * 10**m_order
	return xstep

""" x_axis_info:
		0: xticklabels (if percentage, should be 'percentage';
						if normal ticks, 'normal')
		1: xlabel
		2: total # (for percentages)
		3+: no necessary order unless otherwise stated.
			'print xx' prints the cts/xx;
			'print pp' prints the percentages;
			'print x&p' prints both;       """

def get_x_graph_info(xx,x_axis_info):
	xx = array(xx); cts = xx
	if x_axis_info[0] == 'manual':
		xmin,xmax,xstep,x_ticks,xticklabels = x_axis_info[2]
		return xx,cts,xmin,xmax,xstep,x_ticks,xticklabels
	if 'percentage' in x_axis_info:
		T = x_axis_info[2] # T can be a scalar or an array
		#if is_it(T):
		#	T = T[::-1]
		xx = cts / T
	#if xx.ndim > 1 and 'stacked' not in x_axis_info: # side by side (vertically)
	#	xx = xx[:,::-1]
	#	cts = cts[:,::-1]
	#	if 
	if xx.ndim > 1 and 'stacked' in x_axis_info: # stacked horizontally
		max_x_value = xx.sum(axis=1).max()
	else:
		max_x_value = xx.max()
	xstep = get_xstep(max_x_value)
	xmax = ceil(max_x_value*1.1/xstep)*xstep
	#print(x_axis_info)
	xmax_manual = [xinfo[1] for xinfo in x_axis_info
					if (is_it(xinfo) and xinfo[0]=='Xmax:')]
	if xmax_manual != []:      #  we need this so that we can get
		xmax = xmax_manual[0]  #  room for legends on certain graphs
	if 'percentage' in x_axis_info:
		xmax = min(xmax,1)
	xmin = 0
	x_ticks = arange(xmin,xmax+xstep/2,xstep)
	if x_axis_info[0] == 'percentage':
		xticklabels = [str(int(x*100+.01))+'%' for x in x_ticks]
		if len(xticklabels)!=len(set(xticklabels)):
			xticklabels = [str(x*100)[:3]+'%' for x in x_ticks] # might need fixing
	elif x_axis_info[0] == 'normal':
		xticklabels = x_ticks
	else:
		xticklabels = x_axis_info[0]
	return xx,cts,xmin,xmax,xstep,x_ticks,xticklabels

def get_bar_info(xx,x_axis_info,q_idx,y_ticks,bar_width,bci,cts):
	Nbars = 1 if xx.ndim==1 else xx.shape[1]
	#### #### #### ####
	color_assignment = {}
	for i in range(Nbars):
		if (q_idx,i+1) in bci:
			color_assignment[i] = bci[(q_idx,i+1)]
		elif (0,i+1) in bci:
			color_assignment[i] = bci[(0,i+1)]
		elif (q_idx,0) in bci:
			color_assignment[i] = bci[(q_idx,0)]
		elif q_idx in bci:
			color_assignment[i] = bci[q_idx]
		else:
			color_assignment[i] = bci[0]
	x_info_color = [xinfo[1] for xinfo in x_axis_info
					if (is_it(xinfo) and xinfo[0]=='Color:')]
	if x_info_color != []:
		for i,c in enumerate(x_info_color[0]):
			if 'override' not in bci or (
				0 not in bci['override'] and q_idx not in bci['override']):
				color_assignment[i] = c
				#print(i,c)
	#### #### #### ####
	if Nbars==1:
		barinfo = [dict(left=0,width=xx,
			bottom=y_ticks-bar_width/2,height=bar_width,
			orientation='horizontal',alpha=.8,
			color=color_assignment[0],edgecolor=color_assignment[0])]
		textinfo = [(xx,cts,y_ticks)]
	elif 'stacked' in x_axis_info:
		cum_xx = xx.cumsum(axis=1)
		barinfo = [dict(left=cum_xx[:,i]-xx[:,i],width=xx[:,i],
			bottom=y_ticks-bar_width/2,height=bar_width,
			orientation='horizontal',alpha=.8,
			color=color_assignment[i],edgecolor=color_assignment[i])
					for i in range(Nbars)]
		# note that for a stacked barchart there is a longer textinfo
		textinfo = [(cum_xx[:,i]-xx[:,i],cum_xx[:,i],cts[:,i],y_ticks)
					for i in range(Nbars)]
	elif 'iqr' in x_axis_info:
		#      - - - boxplot-like interquartile range + median - - -
		# sort of lame in that the width of the median shading needs to be 
		# entered (via lower and upper "bounds" of the median).  
		left_edge = xx[:,0]
		med_lower = xx[:,1]
		med_upper = xx[:,2]
		right_edge = xx[:,3]
		barinfo = [
			dict(left=left_edge,width=right_edge-left_edge,#med_lower-left_edge,
			bottom=y_ticks-bar_width/2,height=bar_width,
			orientation='horizontal',alpha=.7,
			color=color_assignment[0],edgecolor=color_assignment[0]),
			dict(left=med_lower,width=med_upper-med_lower,
			bottom=y_ticks-bar_width/2,height=bar_width,
			orientation='horizontal',alpha=1,
			color=color_assignment[1],edgecolor='none'),
			#dict(left=med_upper,width=right_edge-med_upper,
			#bottom=y_ticks-bar_width/2,height=bar_width,
			#orientation='horizontal',alpha=.7,
			#color=color_assignment[0],edgecolor='none'),
						]
		textinfo = [(left_edge,cts,y_ticks),
					(med_upper,cts,y_ticks),
					(right_edge,cts,y_ticks),]
	elif 'iqr alpha' in x_axis_info:
		# same as above but with variable alpha, which is given in xx
		left_edge = xx[:,0]
		med_lower = xx[:,1]
		med_upper = xx[:,2]
		right_edge = xx[:,3]
		bar_alpha = xx[:,4]
		barinfo = [
			dict(left=left_edge[j],width=right_edge[j]-left_edge[j],#med_lower-left_edge,
			bottom=y_ticks[j]-bar_width/2,height=bar_width,
			orientation='horizontal',alpha=bar_alpha[j],
			color=color_assignment[0],edgecolor=color_assignment[0])
						for j in range(len(bar_alpha)) ] + [
			dict(left=med_lower,width=med_upper-med_lower,
			bottom=y_ticks-bar_width/2,height=bar_width,
			orientation='horizontal',alpha=1,
			color=color_assignment[1],edgecolor='k'),
						] 
		textinfo = [(left_edge,cts,y_ticks),
					(med_upper,cts,y_ticks),
					(right_edge,cts,y_ticks),]
	else: # adjacent:
		subbar_bfr_p = .2
		sub_bw = bar_width / (Nbars+Nbars*subbar_bfr_p-subbar_bfr_p)
		subbar_bfr = subbar_bfr_p * sub_bw
		subbar_bottoms = [y_ticks + bar_width/2 - sub_bw*(i+1) - subbar_bfr*i
														for i in range(Nbars)]
		barinfo = [dict(left=0,width=xx[:,i],
			bottom=subbar_bottoms[i],height=sub_bw,
			orientation='horizontal',alpha=.8,
			color=color_assignment[i],edgecolor=color_assignment[i])
					for i in range(Nbars)]
		textinfo = [(xx[:,i],cts[:,i],
					subbar_bottoms[i] + sub_bw/2) for i in range(Nbars)]
	return barinfo,textinfo,color_assignment

def get_int_perc(p,d=0):
	if p==0:
		return '0'
	elif p==1:
		return '100%'
	pstr1 = str(int(round(float(p)*100,d)))
	if pstr1 == '0':
		return '<1%'
	elif pstr1 == '100':
		return '~100%'
	else:
		return pstr1+'%'

def get_legend(legend_info,bar_width,color_assignment,xy_scale,xmin,xmax,ymin,ymax):
	# x,y of legend's bottom right corner specified by values in legend_info.
	# The values correspond to the figure coordinate distances from the lower
	# right corner of the subplot.  If the negative values will take the
	# distance from the other side to the other edge of the box (i.e., left or top).  
	legend_texts = legend_info[3]
	max_str_len = max([measure_string(s,cw_dict=char_width12b)
				for lt in legend_texts for s in lt.split('\n')]) * xy_scale[0]
	max_str_hgt = max([measure_string_hgt(lt,cw_dict=char_width12b)
										for lt in legend_texts]) * xy_scale[1]
	c12 = char_width12b['char_hgt'] * xy_scale[1]
	s12 = char_width12b['spc_hgt'] * xy_scale[1]
	bar_example_width = .2 * xy_scale[0]
	#bar_example_height = max(c12,bar_width/len(legend_texts) *xy_scale[1])
	bar_example_height = c12 * 1.5 * xy_scale[1]
	line_height = max(bar_example_height,max_str_hgt+s12) # should be >= bar_example_height and >=c12
	bfr = .06 # legend_buffer
	xbfr = bfr*xy_scale[0]
	ybfr = bfr*xy_scale[1]
	legend_width = xbfr*3 + bar_example_width + max_str_len
	legend_height = ybfr + (ybfr+line_height) * len(legend_texts)

	d_x = legend_info[1]
	d_y = legend_info[2]
	
	if d_x >= 0:
		SE_x = xmax - d_x * xy_scale[0]
	else:
		SE_x = xmin - d_x * xy_scale[0] + legend_width
	if d_y >= 0:
		SE_y = ymin + d_y * xy_scale[1]
	else:
		SE_y = ymax + d_y * xy_scale[1] - legend_height
	
	leg_text_info = [ [ SE_x - legend_width + xbfr*2 + bar_example_width,
						SE_y + legend_height- ybfr*(i+1) - line_height*(i+.5),
						lt ] for i,lt in enumerate(legend_texts)]
	leg_bar_info = [dict(
			left= SE_x - legend_width + xbfr,
			width= bar_example_width,
			bottom= SE_y + legend_height - (i+1)*(ybfr+line_height)
								+ (line_height-bar_example_height)/2,
			height= bar_example_height,
			orientation='horizontal',alpha=.8,
			color=color_assignment[i],edgecolor=color_assignment[i])
						for i in range(len(legend_texts))]
	leg_bckgrnd_info = dict(
			left= SE_x - legend_width,
			width= legend_width,
			bottom= SE_y,
			height= legend_height,
			orientation='horizontal',alpha=1,
			color=(.99,.99,.99),edgecolor=(.6,.6,.6))
	return leg_bar_info, leg_text_info, leg_bckgrnd_info

def make_bar_subplot(graph_info,ax,q_idx,bci,xy_fig):
	# y ticks & labels #
	ymin,ymax,y_ticks,bar_width = graph_info['y_axis_info']
	ax.set_ylim( ymin, ymax )
	ax.set_yticks( y_ticks )
	y_labels = graph_info['ylabels'][::-1]
	ax.set_yticklabels( y_labels,fontproperties=font12, position=(-.013,0) )
	# x ticks & labels #
	x_axis_info = graph_info['x_axis_info']
	xx = graph_info['xx'][::-1]
	if x_axis_info[0] == 'defined':
		x_ticks,xticklabels,xlabel,xmin,xmax,xstep = x_axis_info[1:7]
		cts = xx
		xx = array(xx)
	else:
		xlabel = x_axis_info[1]
		xx,cts,xmin,xmax,xstep,x_ticks,xticklabels = get_x_graph_info(
						xx,x_axis_info)
	ax.set_xlim(xmin, xmax)
	# xy_scale only used (so far) for the legend. xy_scale[1] should be 1
	xy_scale = ((xmax-xmin)/(xy_fig[2]-xy_fig[0]),
				(ymax-ymin)/(xy_fig[3]-xy_fig[1]))
	ax.set_xticks(x_ticks)
	ax.set_xticklabels(xticklabels,fontproperties=font12)
	ax.set_xlabel(xlabel, fontproperties=font12)
	ax.tick_params(axis='both',left='off',right='off',top='off',bottom='off')
	make_background_bars(ax,xmin,xmax,xstep,x_ticks,ymin,ymax)
	for spine in ax.spines.values():
		spine.set_visible(False)
	barinfo,textinfo,color_assignment = get_bar_info(xx,x_axis_info,q_idx,
												y_ticks,bar_width,bci,cts)
	for bar_i in barinfo:
		ax.bar(**bar_i)
	for text_i in textinfo:
		if 'stacked' in x_axis_info and 'no print' not in x_axis_info:
			# was 'print p&x1 pt8 stackfit', but stacked charts need diff
			# treatment regardless.  text_i is now of len=4.
			for x0,x1,ct,y in zip(*text_i):
				pstr = get_int_perc(x1-x0)
				if pstr!='0': pstr += ', '+str(int(ct))
				slen = measure_string(pstr+' ',cw_dict=char_width8b) * xy_scale[0]
				if slen > x1 - x0:
					pstr = str(int(ct))
					slen = measure_string(pstr+' ',cw_dict=char_width8b) * xy_scale[0]
					if slen > x1 - x0:
						pstr = ''
				ax.text((x0+x1)/2,y,pstr,color=(.3,.3,.3),
						alpha=.7,ha='center',va='center',fontproperties=font8)
		elif 'print cts' in x_axis_info: # note if not perc, this == xx
			for x,ct,y in zip(*text_i):
				ax.text(x+xmax/100,y,'('+str(ct)+')',color=(.5,.5,.5),
						alpha=.6,ha='left',va='center',fontproperties=font10)
		elif 'print xx' in x_axis_info or 'print pp' in x_axis_info:
			for x,ct,y in zip(*text_i):
				ax.text(x+xmax/100,y,'('+str(round(x*100,1))[:4]+'%)',color=(.5,.5,.5),
						alpha=.6,ha='left',va='center',fontproperties=font10)
		elif 'print x&p' in x_axis_info:
			for x,ct,y in zip(*text_i):
				ax.text(x+xmax/100,y,
						'('+str(ct)+':'+str(round(x*100,1))[:4]+'%)',
						color=(.5,.5,.5),
						alpha=.6,ha='left',va='center',fontproperties=font10)
		elif 'print pp1' in x_axis_info:
			for x,ct,y in zip(*text_i):
				pstr = get_int_perc(x)
				ax.text(x+xmax/100,y,'('+pstr+')',color=(.5,.5,.5),
						alpha=.6,ha='left',va='center',fontproperties=font10)
		elif 'print pp1 pt8' in x_axis_info:
			for x,ct,y in zip(*text_i):
				pstr = get_int_perc(x)
				ax.text(x+xmax/100,y,''+pstr,color=(.5,.5,.5),
						alpha=.6,ha='left',va='center',fontproperties=font8)
		elif 'print p&x1 pt8' in x_axis_info:
			for x,ct,y in zip(*text_i):
				pstr = get_int_perc(x)
				if pstr!='0': pstr = pstr +', '+str(int(ct))
				ax.text(x+xmax/100,y,pstr,color=(.5,.5,.5),
						alpha=.6,ha='left',va='center',fontproperties=font8)

	legend_info = [xinfo for xinfo in x_axis_info # this isn't really 'x' info..
					if (is_it(xinfo) and xinfo[0]=='Legend:')]
	if legend_info != [] and 'no legend' not in x_axis_info:
		leg_bar_info,leg_text_info,leg_bckgrnd_info = get_legend(legend_info[0],
						bar_width,color_assignment,xy_scale,xmin,xmax,ymin,ymax)
		ax.bar(**leg_bckgrnd_info)
		for lbi in leg_bar_info:
			ax.bar(**lbi)
		for x,y,t in leg_text_info:
			ax.text(x,y,t,fontproperties=font12,ha='left',va='center')
		




""" Note on bar coloring: barcolor_input (a.k.a. bci) should be a dictionary
	(if it isn't it will be made into one) with keys corresponding to the 
	q_idx (graphs).  0 is the default value.  If there are multiple bars
	in a single graph, the colors can be specified by a tuple key:
	(q_idx,bar_idx) with bar indices starting at 1 and 
	in the order that they appear in graph_data0.  Bar colors can also
	be specified in the graph_data0 as a list/tuple in x_axis_info (in any
	position) whose first item is 'Color:'.  Subsequent items in the list
	will be the colors assigned to the bars (if there aren't enough then the
	default is used).  The bci can 'override' this input if it has a k-v
	with key='override' and the values as the q_idx.  If 0 is in the 'override'
	value then ALL bci preferences will be used (including the defaults) over
	the x_axis_info colors.  Thus override can be customized for only certain
	q_idx; it may not, however, be customized within a q_idx for certain bars.
"""



def generate_bar_graph_pages(
	graph_data0,
	barcolor_input=def_colors1,
	bci=None,
	fig_width=8,
	pg_len_limit=None,
	filename='test_graphs',
	close_figures=True,
	save_opt='pdfpages',
	graph_param_input={},
	print_page_numbers=False,
	print_graph_numbers=False,
	bar_width_function=multiply_by(.7),
	extra_label_space = 'auto',            ):
	# preliminary color stuff:
	if bci!=None:
		barcolor_input = bci
	if is_it(barcolor_input):
		barcolor_input = {0:barcolor_input}
	elif type(barcolor_input)!=dict:
		barcolor_input = {0:[.7,.7,.99]}
	elif 0 not in barcolor_input:
		barcolor_input[0] = [.7,.7,.99]
	for i in barcolor_input:
		if type(barcolor_input[i])==str:
			if barcolor_input[i] in master_color_dict:
				barcolor_input[i] = master_color_dict[barcolor_input[i]]
			else:
				print('unrecognized color string:',barcolor_input[i])
				barcolor_input[i] = [.7,.7,.99]
	#print(barcolor_input)
	# graph_info_dict: this is where we will be putting most of our info
	GID,graph_page_data = make_GID(graph_data0,pg_len_limit,
										extra_label_space=extra_label_space,
										print_continued=True,
										bar_width_function=bar_width_function,
										print_graph_numbers=print_graph_numbers,
										gpi=graph_param_input)
	if save_opt=='pdfpages':
		pp = PdfPages(filename+'.pdf')
	for page_idx,(fig_title,q_indices,fig_hgt) in sorted(graph_page_data.items()):
		N_subplots = len(q_indices)
		current_y = fig_hgt - GID[q_indices[0]]['header_buffer']
		xy_by_ax = []
		subplot_title_tops = []
		for q_idx in q_indices:
			x0 = GID[q_idx]['max_lab_len'] + .2
			x1 = fig_width - .3
			subplot_title_tops.append(current_y - .1)
			y1 = current_y - GID[q_idx]['super_buffer']
			y0 = y1 - GID[q_idx]['subplot_hgt']
			current_y = y0 - GID[q_idx]['sub_buffer']
			xy_by_ax.append((x0,y0,x1,y1))
			if x0 > x1 or y0 > y1:
				print('bad coordinates for',q_idx,':')
				print('  ',x0,y0,x1,y1)
				print(GID[q_idx])
		fig,axx = make_custom_figure((fig_width,fig_hgt),xy_by_ax)
		if fig_title != '':
			fig.text(.15/fig_width,1-.15/fig_hgt,fig_title,
					fontproperties=font18,ha='left',va='top')
		if print_page_numbers:
			fig.text(1-.15/fig_width,1-.15/fig_hgt,'(page '+str(page_idx)+')',
					fontproperties=font12,ha='right',va='top')
		for q_idx,ax,subtit_top,xy_fig in zip(
								q_indices,axx,subplot_title_tops,xy_by_ax):
			make_bar_subplot(GID[q_idx],ax,q_idx,barcolor_input,xy_fig)
			# acutally a figure text, not subplot text:
			if print_graph_numbers:
				subtit_string = '('+str(q_idx)+')  '+GID[q_idx]['q_name']
			else:
				subtit_string = GID[q_idx]['q_name']
			if GID[q_idx]['q_name']!='':
				fig.text(.5/fig_width,subtit_top/fig_hgt,subtit_string,
					fontproperties=font14, ha='left', va='top')
		if save_opt=='pdfpages':
			fig.savefig(pp, format='pdf')
		elif save_opt=='png':
			fig.savefig(filename+'.png',dpi=300)
		if close_figures:
			plt.close(fig)
	if save_opt=='pdfpages':
		pp.close()


def generate_bar_graph_sepfiles(graph_data0,**kwargs):
	for kw in 'print_page_numbers','print_graph_numbers':
		if kw not in kwargs:
			kwargs[kw] = False
	filename_base = kwargs['filename']
	for group_name,group_data in graph_data0:
		for q_data in group_data:
			graph_data1 = [[group_name,[q_data]]]
			fn_add = q_data[0].lower().replace(' ','_').replace('/','-')
			kwargs['filename'] = filename_base+'_'+fn_add
			kwargs['save_opt'] = 'png'
			generate_bar_graph_pages(graph_data1,**kwargs)


def break_line(t,max_width,cw_dict):
	if measure_string(t,cw_dict=cw_dict) < max_width:
		return t
	new_t_list = ['']
	for w in t.split():
		wlen = measure_string(w,cw_dict=cw_dict)
		lastline_len = measure_string(new_t_list[-1]+' ',cw_dict=cw_dict)
		if wlen > max_width:
			print('Problem: single word too long! -->',w)
			return t
		if new_t_list[-1] == '':
			new_t_list[-1] += w
		elif lastline_len + wlen > max_width:
			new_t_list.append(w)
		else:
			new_t_list[-1] += ' ' + w
	return '\n'.join(new_t_list)

def phi_coef(a,b,c,d):
	return (a*d - b*c) / ((a+b)*(a+c)*(b+d)*(c+d))**.5

c12 = char_width12b['char_hgt']
s12 = char_width12b['spc_hgt']
c14 = char_width14b['char_hgt']
s14 = char_width14b['spc_hgt']

def correlation_square_plot(
	graph_data0,
	filename='corrtest',
	closefig=True,
	max_x_sq=3,
	max_y_sq=None,
	full_sq_N=800,
	fig_width=8,            ):
	gr_name,gr_data = graph_data0
	left_margin = .2 + c12
	#left_margin = .2
	right_margin = .2
	axx_xys = []
	text_locations = []
	xborders = linspace(0,fig_width,max_x_sq+1)
	sq_xx = [(x0+left_margin,x1-right_margin)
				for x0,x1 in zip(xborders[:-1],xborders[1:])]
	sq_width = sq_xx[0][1] - sq_xx[0][0]
	linebreak_dict = {}
	for ab_abbr,(a_full,b_full),ab_nn in gr_data:
		a_full_new = break_line(a_full,sq_width,char_width12b)
		if a_full_new != a_full:
			#print([a_full_new,a_full,sq_width])
			linebreak_dict[a_full] = a_full_new
		b_full_new = break_line(b_full,sq_width,char_width12b)
		if b_full_new != b_full:
			linebreak_dict[b_full] = b_full_new
	# note that we start with fig_hgt = 0, then will reset all of these #s
	current_y = 0 - .2 - char_width18b['char_hgt'] - char_width18b['spc_hgt']
	for j0 in arange(0,len(gr_data),max_x_sq):
		maxlines = max([ sum([  t.count('\n')
								if t not in linebreak_dict
								else linebreak_dict[t].count('\n')
								for t in (a_full,b_full)   ])
						for ab_abbr,(a_full,b_full),ab_nn
						in gr_data[j0:j0+max_x_sq]  ])
		text_hgt = ( c14
					+s14 * 2
					+c12 * (maxlines+2)
					+s12 * (maxlines+5)  )
		current_y -= text_hgt
		for j in arange(j0,min(len(gr_data),j0+max_x_sq)):
			x0,x1 = sq_xx[j%max_x_sq]
			a_full,b_full = gr_data[j][1]
			if a_full in linebreak_dict:
				y2_Nl = linebreak_dict[a_full].count('\n') + 1
			else: 
				y2_Nl = a_full.count('\n') + 1
			if b_full in linebreak_dict:
				y3_Nl = linebreak_dict[b_full].count('\n') + 1
			else: 
				y3_Nl = b_full.count('\n') + 1
			y4_ctr = current_y + s12 + c12/2
			y3_hgt = y3_Nl * c12 + (y3_Nl-1) * s12
			y3_ctr = y4_ctr + c12/2 + s12 + y3_hgt/2
			y2_hgt = y2_Nl * c12 +(y2_Nl-1) * s12
			y2_ctr = y3_ctr + y3_hgt/2 + s12 + y2_hgt/2
			y1_ctr = y2_ctr + y2_hgt/2 + s14 + c14/2
			text_locations.append([(x0,y1_ctr),(x0,y2_ctr),
									(x0,y3_ctr),(x0,y4_ctr)])
			axx_xys.append([x0,current_y-sq_width,x1,current_y])
		current_y -= (sq_width  + c12 + s12*3)
	fig_hgt = -current_y
	axx_xys = [[x0,y0+fig_hgt,x1,y1+fig_hgt] for (x0,y0,x1,y1) in axx_xys]
	text_locations = [[(x,y+fig_hgt) for x,y in xys] for xys in text_locations]
	#######
	fig,axx = make_custom_figure((fig_width,fig_hgt),axx_xys)
	#######
	fig.text(.15/fig_width,1-.15/fig_hgt,gr_name,
				fontproperties=font18,ha='left',va='top')
	for j in range(len(gr_data)):
		(a_abbr,b_abbr),(a_full,b_full),ab_nn = gr_data[j]
		N = sum(ab_nn)
		expd_vals = [(ab_nn[0]+ab_nn[1]) * (ab_nn[0]+ab_nn[2]) / N,
					(ab_nn[0]+ab_nn[1]) * (ab_nn[1]+ab_nn[3]) / N,
					(ab_nn[0]+ab_nn[2]) * (ab_nn[2]+ab_nn[3]) / N,
					(ab_nn[1]+ab_nn[3]) * (ab_nn[2]+ab_nn[3]) / N ]
		chisq_cmpnts = [(n-ev)**2/ev for n,ev in zip(ab_nn,expd_vals)]
		q_chisq = sum(chisq_cmpnts)
		phi = phi_coef(*ab_nn)
		p_q = 1 - ss.chi2.cdf(q_chisq,1,1)
		colors = (0,0,0),(.2,.4,.2),(.4,.2,.4),(0,0,0)
		for t,(x,y),c,font_pt in zip(
							[   a_abbr+' vs '+b_abbr,
							    a_full,
							    b_full,
							    'phi='+str(phi)[:5]+
							    '; Q='+str(q_chisq)[:5] ],  
							    # + ' (p='+str(p_q)[:4]+')'  ],
							text_locations[j],
							colors,
							[14]+[12]*3,  ):
			fnt = {14:font14,12:font12}[font_pt]
			if t in linebreak_dict and font_pt==12:
				t = linebreak_dict[t]
			fig.text(x/fig_width,y/fig_hgt,t,color=c,
				fontproperties=fnt,ha='left',va='center')
		ax = axx[j]
		ax.broken_barh([[0,1]],[0,1],color=(.7,.7,.7),alpha=.1)
		ax.set_xlim(0,1); ax.set_ylim(0,1)
		cell_centers = [(.75,.75),(.25,.75),(.75,.25),(.25,.25)]
		cell_labels = ['Both',a_abbr,b_abbr,'Neither']
		for (x,y),t,n,ev in zip(cell_centers,cell_labels,ab_nn,expd_vals):
			emphasis = min((n-ev)**2/ev,7) # chisq component with ceiling
			off_color = .75-emphasis/10
			if n > ev:
				c = (.75,off_color,off_color)
			else:
				c = (off_color,off_color,.75)
			cell_side = min( max(n,1) / full_sq_N , 1 ) ** .25 / 2
			ax.broken_barh([[x-cell_side/2,cell_side]],
							[y-cell_side/2,cell_side],color=c,alpha=.4)
			##### ##### ##### ##### ##### ##### ##### #####
			cs_fig = sq_width * cell_side # cell_side in figure coords
			t_len = measure_string(t+'  ',cw_dict=char_width12b)
			n_len = measure_string(str(n)+'  ',cw_dict=char_width14b)
			#t_hgt = c12+2*s12; n_hgt = c14+2*s14
			t_hgt = c12+s12; n_hgt = c14+s14
			if cs_fig > t_hgt+n_hgt+s12+s14 and cs_fig > max(t_len,n_len):
				ax.text(x,y+s12/sq_width,t,
							fontproperties=font12,ha='center',va='bottom')
				ax.text(x,y-s14/sq_width,n,
							fontproperties=font14,ha='center',va='top')
			elif cs_fig > t_hgt+n_hgt and cs_fig > max(t_len,n_len):
				ax.text(x,y,t,fontproperties=font12,ha='center',va='bottom')
				ax.text(x,y,n,fontproperties=font14,ha='center',va='top')
			elif cs_fig > n_hgt and cs_fig > n_len:
				ax.text(x,y+cell_side/2,t,
						fontproperties=font12,ha='center',va='bottom')
				ax.text(x,y,n,fontproperties=font14,ha='center',va='center')
			else:
				ax.text(x,y+cell_side/2,t,
						fontproperties=font12,ha='center',va='bottom')
				ax.text(x,y-cell_side/2-s14,n,
						fontproperties=font14,ha='center',va='center')
		ax.set_xticks((.25,.75))
		ax.set_yticks((.25,.75))
		ax.set_xticklabels((ab_nn[3]+ab_nn[1],ab_nn[2]+ab_nn[0]),
									fontproperties=font12)
		ax.set_yticklabels((ab_nn[3]+ab_nn[2],ab_nn[1]+ab_nn[0]),
									fontproperties=font12,rotation=90)
		ax.tick_params(axis='both',left='off',right='off',top='off',bottom='off')
		for spine in ax.spines.values():
			spine.set_visible(False)
	fig.savefig(filename+'.pdf')
	if closefig==True:
		plt.close(fig)

################################################################################
################################################################################
####
####
####def generate_test_data():
####	test_data = [
####		[
####			'Group A', [
####				[	'Graph I',
####					['percentage','Share of Respondents (T=50)',50,'print xx'],
####					[[lab,int(x)] for lab,x in zip('abcdefgh',10+(3-arange(8)/3)**3)]
####					],
####				[	'Graph II',
####					['percentage','Share of Respondents (T=60)',60,'print xx'],
####					[[lab*3,x] if lab!='c' else [lab*5+'\n'+(lab*5).upper(),x]
####											for lab,x in zip('abcdef',10+arange(6)**2)]
####					],
####				[	'Graph III',
####					['percentage','Share of Respondents (T=100)',100,'print xx'],
####					[[(lab*2).title()+'\n'[:randint(7)//4]+(lab*3).title(),randint(20)+2]
####																for lab in 'abcdefghijk']
####					], ]
####			],
####		[
####			'Group B', [
####				[	'Graph IV',
####					['percentage','Share of Respondents (T=1000)',1000,'print xx'],
####					[['\n'.join([lab.upper()*6]*3),10+randint(100)] for lab in 'abc']
####					],
####				[	'Graph V',
####					['percentage','Share of Respondents (T=11000)',11000,'print xx'],
####					[[lab.upper()*15,500+randint(100)**2] if lab!='f' else
####					['\n'.join([lab*15]*3),500+randint(100)**2] for lab in 'abcdefghijkln']
####					], ]
####			],
####		[
####			'Group C', [
####				[	'Graph VI',
####					['percentage','Share of Respondents (T=1000)',1000,'print xx'],
####					[['\n'.join([lab.upper()*6]*4),10+randint(100)] for lab in 'abc']
####					],
####				[	'Graph VII',
####					['percentage','Share of Respondents (T=11000)',11000,'print xx'],
####					[['\n'.join([lab*15]*2),500+randint(100)**2] if lab!='f' else
####					['\n'.join([lab*15]*3),500+randint(100)**2] for lab in 'abcdefghijkln']
####					], ]
####			],
####		]
####	return test_data
####
####
################################################################################
#    for testing make_custom_figure( ):
#
#  4  |-----------------------|
#     |                       |
#  3  |   +---+               |
#     |   |   |   +-------+   |
#  2  |   +---+   |       |   |
#     | +---+     |       |   |
#  1  | |   |     +-------+   |
#     | +---+                 |
#  0  |_______________________|
#
#     0   1   2   3   4   5   6
#
#   #   xys = [(.5,.5,1.5,1.5),(1,2,2,3),(3,1,5,2.5),(4,3,5.9,3.9)]
#   #   fig,axx = make_custom_figure((6,4),xys)
#   #   for i,((x0,y0,x1,y1),ax) in enumerate(zip(xys,axx)):
#   #   	ax.set_xlim(x0,x1)
#   #   	ax.set_ylim(y0,y1)
#   #   	ax.plot([x0,(2*x0+x1)/3,x1,(x0+2*x1)/3,x0],
#   #   			[(2*y0+y1)/3,y1,(y0+2*y1)/3,y0,(2*y0+y1)/3],
#   #   			'-o',lw=5,
#   #   			color=(.8*i/(len(xys)-1)+.1,.2,.9-.8*i/(len(xys)-1)) )
#   #
# # # # # --- for testing character dicts:
#   #
#   #   xys = [(1,1,7,7)]
#   #   fig,axx = make_custom_figure((8,8),xys)
#   #   ax = axx[0]; ax.set_xlim(1,7); ax.set_ylim(1,7)
#   #   ax.plot([2,2,6,6,2],[2,6,6,2,2],'b-')
#   #   for d in (.1,.2,.3):
#   #   	ax.plot([2-d,2-d,6+d,6+d,2-d],[2-d,6+d,6+d,2-d,2-d],'r-')
#   #   tt = [str(i//10)+str(i%10)
#   #   	+'345678901234567890123456789012345678901234567890'
#   #   							for i in range(1,31)]
#   #   ax.text(4,4,'\n'.join(tt),ha='center',va='center',
#   #   	fontproperties=font12)
#   #   
# # # # # # # # # #
#   #
#   #   s0 = 'ABCDEFGHIJKLMNOPQR'
#   #   c = 'u'
#   #   s = s0 + ' ' + s0.lower() + ' ' + s0
#   #   s2 = s +'W 50'
#   #   nlines = 20
#   #   S = '\n'.join([s2]*nlines)
#   #   chgt = char_width12b['char_hgt']
#   #   shgt = char_width12b['spc_hgt']
#   #   chgt = char_width18b['char_hgt']
#   #   shgt = char_width18b['spc_hgt']
#   #   y0_calc = 11-(nlines*chgt+(nlines-1)*shgt)
#   #   y0_manl = 7.2
#   #   slen = measure_string(s,cw_dict=char_width18b)
#   #   fig,axx = make_custom_figure((12,12),[(1,y0_calc,1+slen,11)])
#   #   axx[0].set_xlim(1,1+slen)
#   #   axx[0].set_ylim(11-nlines*chgt-(nlines-1)*shgt,11)
#   #   axx[0].text(1,11.0,S,ha='left',va='top',fontproperties=font18)
#   #   yy1 = 11 - arange(nlines)*(chgt+shgt)
#   #   yy2 = yy1 - chgt
#   #   for yy in (yy1,yy2):
#   #   	for y in yy:
#   #   		axx[0].plot([1,1+slen],[y,y],'r-')
#   #   savefig('chartest.pdf')
#   #   close()
#   #   
#   #   y_fig = 10.2
#   #   x_fig = 8
#   #   gs = gridspec.GridSpec(1, 1,
#   #   	bottom = 1 - 7.5/y_fig,
#   #   	top =    1 - 6.7/y_fig,
#   #   	left =   .2/x_fig,
#   #   	right =  (.2+2.351)/x_fig    )[0]
#   #   fig.add_subplot(gs)
#   #   
################################################################################


#   #   csqp_testdata0 = ['Test Correlations',[
#   #   	[('X','Y'),('XxX','YyY'),[
#   #   		230, # in both
#   #   		114, # in X not Y
#   #   		193, # in Y not X
#   #   		269, # in neither
#   #   		]] ] +  [[(x,y),('\n'.join([x+x.lower()+x]*(1+randint(3))),
#   #   						'\n'.join([y+y.lower()+y]*2)),
#   #   				[randint(400)+10 for i in range(4)]] for
#   #   					x in 'ABC' for y in 'DEF'] ]
#   #   




# # # # # # FOR TESTING:
# # #
#   #   test_data1 = generate_test_data()
#   #   test_data2 = generate_test_data()
#   #   
#   #   # test_graph_param_input1={'left_margin':{0:.2,1:.08,2:.23,4:.15} }
#   #   # test_graph_param_input2={'left_margin':{0:.2,1:.08,2:.23,4:.15} }
#   #   
#   #   test_barcolor_input = {0:(.8,.3,.1),1:(.1,.1,.7),3:(.3,.4,.9),5:(.2,.8,.1)}
#   #   
#   #   
#   #   generate_bar_graph_pages(test_data1,
#   #   			filename='test_graphs1',
#   #   			pg_len_limit=None,
#   #   			barcolor_input=test_barcolor_input,
#   #   			close_figures=True,
#   #   			#graph_param_input=test_graph_param_input1,
#   #   															)
#   #   generate_bar_graph_pages(test_data2,
#   #   			filename='test_graphs2',
#   #   			pg_len_limit=None,
#   #   			barcolor_input=test_barcolor_input,
#   #   			close_figures=True,
#   #   			#graph_param_input=test_graph_param_input2,
#   #   															)






#   #   def testgraph():
#   #   	lb_list = [0,0,1,1,2,2,0,3,1,3,3,2]
#   #   	labs = ['\n'.join([''.join('ABCDEFGHpqykgj'[randint(12)] 
#   #   						for i in range(8))]*(lb+1)) for lb in lb_list]
#   #   	N_yy = len(labs)
#   #   	left_margin = .6
#   #   	header_buffer = 1.5
#   #   	super_buffer = 1.5
#   #   	sub_buffer = 2.2
#   #   	y_edge_buffer = .3
#   #   	bar_width = .5
#   #   	char_hgt = .479
#   #   	spc_hgt = .234
#   #   	SPACING = 2 # 2 = double spacing
#   #   	lb_tups = [(lb1,lb2) for lb1,lb2 in zip(lb_list[:-1],lb_list[1:])]
#   #   	#### * #### * #### * #### * #### * #### * ####
#   #   	# YIBs = [.6*(lb1+lb2+2)/2 for lb1,lb2 in lb_tups]
#   #   	YIBs = [  (lb1+lb2)/2 * spc_hgt
#   #   			+ (lb1+lb2+2)/2 * char_hgt
#   #   			- bar_width
#   #   			+ spc_hgt * SPACING
#   #   			for lb1,lb2 in lb_tups]
#   #   	#### * #### * #### * #### * #### * #### * ####	
#   #   	subplot_hgt = 2*y_edge_buffer + N_yy * bar_width + sum(YIBs)
#   #   	fig_hgt = subplot_hgt + header_buffer + super_buffer + sub_buffer
#   #   	top_margin = 1 - (header_buffer + super_buffer) / fig_hgt
#   #   	bottom_margin = sub_buffer / fig_hgt
#   #   	right_margin = .96 # should be fine for all cases; if not make this a param
#   #   	print(fig_hgt,top_margin,subplot_hgt,bottom_margin,fig_hgt/4)
#   #   	fig = figure(figsize=(8,fig_hgt/4))
#   #   	fig.suptitle('TEST GRAPH PAGE',x=.03,y=1-.03*20/fig_hgt,
#   #   							fontproperties=font18,ha='left')
#   #   	gs = gridspec.GridSpec(1, 1,
#   #   		bottom=bottom_margin, top=top_margin,
#   #   		left=left_margin, right=right_margin )
#   #   	ax = fig.add_subplot( gs[0] )
#   #   	##################
#   #   	# ticks & labels #
#   #   	xmin = 0; xmax = 100; xstep = 10
#   #   	ax.set_xlim( xmin, xmax )
#   #   	x_ticks = arange(xmin,xmax,xstep)
#   #   	ax.set_xticks( x_ticks )
#   #   	ax.set_xticklabels( [str(x)+'%' for x in x_ticks],
#   #   				fontproperties=font12)
#   #   	ax.set_xlabel('Perecentage', fontproperties=font12)
#   #   	ymin = 0
#   #   	ymax = 2*y_edge_buffer + N_yy*bar_width + sum(YIBs)
#   #   	ax.set_ylim( ymin, ymax )
#   #   	y_ticks = (y_edge_buffer + bar_width/2 + array([0]+YIBs).cumsum() 
#   #   			+ arange(N_yy) * bar_width)[::-1]
#   #   	ax.set_yticks( y_ticks )
#   #   	ax.set_yticklabels( labs[::-1],fontproperties=font12, position=(-.01,0) )
#   #   	ax.tick_params(axis='both',left='off',right='off',top='off',bottom='off')
#   #   	ax.set_title('(1)  Test Graph', 
#   #   		fontproperties=font14, loc='left')
#   #   	##################
#   #   	# bckgrnd & bars #
#   #   	for spine in ax.spines.values():
#   #   		spine.set_visible(False)
#   #   	bckgrnd_bar_midgap = xmax/100
#   #   	bckgrnd_bar_edgegap = 0
#   #   	bb_left_edges = x_ticks[:-1] + bckgrnd_bar_midgap/2
#   #   	bb_widths = [xstep - bckgrnd_bar_midgap] * len(bb_left_edges)
#   #   	bb_left_edges[0] = xmin + bckgrnd_bar_edgegap
#   #   	ax.bar( bottom=ymin,  height=[ymax-ymin]*len(x_ticks[1:]),
#   #   			left=bb_left_edges, width=bb_widths, 
#   #   			color=(0,0,0),alpha=.03,edgecolor=(1,1,1))
#   #   	barcolor = [.4,.2,.9]
#   #   	xx = [randint(90)+5 for lab in labs]
#   #   	#ax.bar( left=0, height=bar_width, width=xx, bottom=y_ticks-bar_width/2,
#   #   	#		color=barcolor,alpha=.8,orientation='horizontal',
#   #   	#		edgecolor=barcolor)
#   #   	#for x,y in zip(xx,y_ticks):
#   #   	#	ax.text(x+xmax/100,y,'('+str(x)+')',color=(.5,.5,.5),
#   #   	#			alpha=.6,ha='left',va='center',fontproperties=font12)
#   #   	pp = PdfPages('testgraph0.pdf')
#   #   	fig.savefig(pp, format='pdf')
#   #   	pp.close()
#   #   	#close(fig)
#   #   	print(fig_hgt,subplot_hgt,(xmin,xmax),(ymin,ymax))
#   #   

# text(65,3,'ABCDEF\nGHIJKL\nMNOPQRS\nTUVXYZ\n'*2,fontproperties=font12)
# for y in (cumsum(concatenate(([0], tile((.479,.234),8)))) + 3):
	# plot([60,80],[y,y],'r-',lw=1,alpha=1)
# savefig('graphtest1.pdf')
# 
# 
# char_hgt = .479
# spc_hgt = .234
# 
# 
# close()
# testgraph()
# 
# for y in arange(2,25,2)-.1:
	# for x in arange(0,100,10):
		# plot([x,x],[y-.6,y-.2],'b-',alpha=.7)
# 
# for y in arange(2,25,2)-.1:
	# for x in arange(0,100,1):
		# plot([x,x],[y-.2,y],'r-',alpha=.5)
# 
# n_chars = 100
# cc = '\'ijl ftI./!,:;r-()"*^cksvxyzJabdeghnopquL0123456789=FTZABEKPSVXYwCDHNRUGOQmM%W@'
# batch = 1
# for y0,c in enumerate(cc[11*batch:11*(batch+1)]):
	# if c!=' ':
		# text(0,22-y0*2,c*(n_chars+1),ha='left',fontproperties=font12)
	# else:
		# text(0,22-y0*2,c*(n_chars)+'...',ha='left',fontproperties=font12)
# 
# 
# savefig('graphtest1.pdf')
# 
# 
# 
# 

# char_width0 = {
	# "'": 0.536,
	# 'ijl': 0.623,
	# ' ftI./!,:;': 0.78,
	# 'r-()': 0.935,
	# '"': 0.996,
	# '*': 1.092,
	# '^': 1.118,
	# 'cksvxyzJ': 1.402,
	# 'abdeghnopquL0123456789#$': 1.56,
	# '=': 1.638,
	# 'FTZ': 1.714,
	# 'ABEKPSVXY&': 1.872,
	# 'wCDHNRU': 2.026,
	# 'GOQ': 2.183,
	# 'mM': 2.337,
	# '%': 2.497,
	# 'W': 2.65,
	# '@': 2.843,
		# }
# 
# char_width1 = {
	# "'":                        0.1303552,
	# 'ijl':                      0.1515136,
	# ' ftI./!,:;':               0.189696,
	# 'r-()':                     0.227392,
	# '"':                        0.2422272,
	# '*':                        0.2655744,
	# '^':                        0.2718976,
	# 'cksvxyzJ':                 0.3409664,
	# 'abdeghnopquL0123456789#$': 0.3793920,
	# '=':                        0.3983616,
	# 'FTZ':                      0.4168448,
	# 'ABEKPSVXY&':               0.4552704,
	# 'wCDHNRU':                  0.4927232,
	# 'GOQ':                      0.5309056,
	# 'mM':                       0.5683584,
	# '%':                        0.6072704,
	# 'W':                        0.6444800,
	# '@':                        0.6914176,   }
# 
# char_width2 = {c:x for cc,x in char_width.items() for c in cc}
# 
# chars = [choice(list(char_width2)) for i in range(1000)]
# 
# lines = []; newline = ''; l = 0
# for c in chars:
	# if c in '\'"%$/;:^=':
		# continue
	# elif char_width2[c] + l <= 16:
		# newline += c
		# l += char_width2[c]
	# else:
		# lines.append(newline)
		# if len(lines)==12:
			# break
		# else:
			# newline = ''; l = 0
# 
# 
# figure(figsize=(8,7.2095))
# ax = subplot(111)
# plot(randn(1000),randn(1000),'go',alpha=.1)
# ymin,ymax = ax.get_ylim()
# ax.set_yticks(linspace(ymin,ymax,14)[1:-1])
# ax.set_yticklabels(lines,fontproperties=font12)
# subplots_adjust(left=.5)
# 
# '\'ijl ftI./!,:;r-()"*^cksvxyzJabdeghnopquL0123456789=FTZABEKPSVXYwCDHNRUGOQmM%W@'






# previous draft, number 1:
#   #   def generate_bar_graph_pages(   graph_data0,
#   #   								barcolor_input=[.4,.2,.9],
#   #   								pg_len_limit=None,
#   #   								filename='test_graphs',
#   #   								close_figures=True,
#   #   								graph_param_input={},
#   #   								text_space_override=True,
#   #   								figtitlefont=font18,
#   #   															):
#   #   	char_width1 = {
#   #   		"'":                        0.1303552,
#   #   		'ijl':                      0.1515136,
#   #   		' ftI./!,:;':               0.189696,
#   #   		'r-()':                     0.227392,
#   #   		'"':                        0.2422272,
#   #   		'*':                        0.2655744,
#   #   		'^':                        0.2718976,
#   #   		'cksvxyzJ':                 0.3409664,
#   #   		'abdeghnopquL0123456789#$': 0.3793920,
#   #   		'=':                        0.3983616,
#   #   		'FTZ':                      0.4168448,
#   #   		'ABEKPSVXY&':               0.4552704,
#   #   		'wCDHNRU':                  0.4927232,
#   #   		'GOQ':                      0.5309056,
#   #   		'mM':                       0.5683584,
#   #   		'%':                        0.6072704,
#   #   		'W':                        0.6444800,
#   #   		'@':                        0.6914176,   }
#   #   	char_width2 = {c:x for cc,x in char_width1.items() for c in cc}
#   #   	char_width2.update({'default':0.4})
#   #   	graph_params = {
#   #   		'left_margin': {0:.2},
#   #   		'header_buffer': {0:1.5},
#   #   		'super_buffer': {0:1.5},
#   #   		'sub_buffer': {0:2.2},
#   #   		'y_edge_buffer': {0:.3},
#   #   		'y_interbar_buffer': {0:.2}, # irrelevant if text_space_override
#   #   		'bar_width': {0:.8},         # irrelevant if text_space_override
#   #   									}
#   #   	graph_params.update({
#   #   		param: param_val if type(param_val)==dict else {0:param_val}
#   #   							for param,param_val in graph_param_input.items()})
#   #   	# graph_info_dict: this is where we will be putting most of our info
#   #   	GID = { q_idx+1: 
#   #   		{   'q_name':q_data[0],
#   #   			'x_axis_info':q_data[1],
#   #   			'xy_data':q_data[2],
#   #   			'group_idx':group_idx,
#   #   			'N_yy':len(q_data[-1])   }
#   #   				for q_idx,(q_data,group_idx) in enumerate(
#   #   				[(q_data,group_idx) for group_idx,(group_name,group_data)
#   #   				in enumerate(graph_data0) for q_data in group_data] ) }
#   #   	for q_idx,graph_info in GID.items():
#   #   		for param,param_val in graph_params.items():
#   #   			if q_idx in param_val:
#   #   				graph_info[param] = param_val[q_idx]
#   #   			else:
#   #   				graph_info[param] = param_val[0]
#   #   		if text_space_override != False:
#   #   			# TSO = text_space_override
#   #   			lb_cts = [lab.count('\n') if type(lab)==str else 0
#   #   							for lab,x in GID[q_idx]['xy_data']]
#   #   			lb_tups = {(l1,l2) for l1,l2 in zip(lb_cts[:-1],lb_cts[1:])}
#   #   			# bw,yib = max([TSO[tup] if type(TSO)==dict else TSO(tup) for tup in tups])
#   #   			#### i think the following parameters are good enough to keep,
#   #   			#### i.e., no customization options necessary.  
#   #   			char_hgt = .479; spc_hgt = .234; extra_spc = 2.5 # (2 means double space)
#   #   			nec_spacing = max( (lb1+lb2)/2 * spc_hgt        # label linebreaks
#   #   								+ (lb1+lb2+2)/2 * char_hgt  # label line (chars)
#   #   								+ spc_hgt * extra_spc       # space btwn labels
#   #   								for lb1,lb2 in lb_tups)
#   #   			bar_to_buffer_ratio = .7 # when labels get further apart, the expansion
#   #   			# is shared in the graph between the bars and the space between the bars.
#   #   			# the b_to_b ratio determines 
#   #   			bar_width = bar_to_buffer_ratio * nec_spacing
#   #   			GID[q_idx]['bar_width'] = bar_width
#   #   			GID[q_idx]['y_interbar_buffer'] = nec_spacing - bar_width
#   #   		graph_info['subplot_hgt'] = (
#   #   				(GID[q_idx]['N_yy']-1) 
#   #   				   * GID[q_idx]['y_interbar_buffer']
#   #   				+ 2 * GID[q_idx]['y_edge_buffer']
#   #   				+ GID[q_idx]['N_yy']
#   #   				   * GID[q_idx]['bar_width']	)
#   #   	########################################
#   #   	# here we distribute our graphs to pages
#   #   	graph_page_data = {} # {page_idx: [figure_title,[q_indices],figure_height]}
#   #   	prev_group_idx = -1 # dummy -1, group_idx starts at 0
#   #   	q_idx = 1; page_idx = 0 # note that q_idx and page_idx BOTH start at 1
#   #   	while q_idx in GID:
#   #   		group_idx = GID[q_idx]['group_idx']
#   #   		add_hgt = sum(GID[q_idx][param] for param in 
#   #   							('subplot_hgt','super_buffer','sub_buffer'))
#   #   		if group_idx != prev_group_idx:
#   #   			page_idx += 1
#   #   			group_name = graph_data0[group_idx][0]
#   #   			graph_page_data[page_idx] = [group_name,[q_idx],
#   #   					sum(GID[q_idx][param] for param in 
#   #   							('subplot_hgt','super_buffer','sub_buffer',
#   #   								'header_buffer'))] # header buffer probably constant
#   #   		elif (pg_len_limit==None or 
#   #   				graph_page_data[page_idx][2] + add_hgt <= pg_len_limit):
#   #   			graph_page_data[page_idx][1].append(q_idx)
#   #   			graph_page_data[page_idx][2] += add_hgt
#   #   		else:
#   #   			page_idx += 1
#   #   			if group_name!=None:
#   #   				group_name = group_name + ' (continued)'
#   #   			graph_page_data[page_idx] = [group_name,[q_idx],
#   #   					sum(GID[q_idx][param] for param in 
#   #   						('subplot_hgt','super_buffer','sub_buffer','header_buffer'))]
#   #   		prev_group_idx = group_idx
#   #   		q_idx += 1
#   #   	pp = PdfPages(filename+'.pdf')
#   #   	for page_idx,(fig_title,q_indices,fig_hgt) in sorted(graph_page_data.items()):
#   #   		N_subplots = len(q_indices)
#   #   		hratios = [GID[q_idx]['subplot_hgt'] for q_idx in q_indices]
#   #   		avg_subplot_hgt = sum(hratios)/N_subplots
#   #   		""" note that only one value is taken for the header/sub-/sup-buffers 
#   #   			on a single page.  the first q_idx is used:                       """
#   #   		header_buffer,super_buffer,sub_buffer = [ GID[min(q_indices)][param]
#   #   					for param in ('header_buffer','super_buffer','sub_buffer')]
#   #   		top_margin = 1 - (header_buffer + super_buffer) / fig_hgt
#   #   		bottom_margin = sub_buffer / fig_hgt
#   #   		height_space = (super_buffer + sub_buffer) / avg_subplot_hgt
#   #   		right_margin = .96 # should be fine for all cases; if not make this a param
#   #   		fig = plt.figure(figsize=(8,fig_hgt/4))
#   #   		fig.suptitle(fig_title+'  ('+str(page_idx)+')',x=.03,y=1-.03*20/fig_hgt,
#   #   								fontproperties=figtitlefont,ha='left')
#   #   		axx = []
#   #   		for q_idx in sorted(q_indices):
#   #   			x_axis_info = GID[q_idx]['x_axis_info']
#   #   			y_labels,xx = zip(*GID[q_idx]['xy_data'][::-1])
#   #   			N_yy = GID[q_idx]['N_yy'] #  = len(y_labels)
#   #   			""" note that left_margin is the sole difference in gridspecs of
#   #   			    subplots on the same page (i.e., in the same figure)         """
#   #   			if text_space_override==False:
#   #   				left_margin = GID[q_idx]['left_margin']
#   #   			else:
#   #   				max_lab_len = max([sum(char_width2[c] if c in char_width2
#   #   								else char_width2['default'] for c in sublab)
#   #   								for lab in y_labels for sublab in str(lab).split('\n')])
#   #   				left_margin = max_lab_len / 32 + .03
#   #   			gs = gridspec.GridSpec(N_subplots, 1,
#   #   				bottom=bottom_margin, top=top_margin,
#   #   				hspace=height_space, height_ratios=hratios,
#   #   				left=left_margin, right=right_margin )
#   #   			ax = fig.add_subplot( gs[q_idx-min(q_indices)] ); axx.append(ax)
#   #   			##################
#   #   			# ticks & labels #
#   #   			if x_axis_info[0] == 'percentage':
#   #   				""" For percs, x_axis_info should be ['percentage', XLABEL,
#   #   				    TOTAL #, 'print xx' (or not - this prints the ct)]       """
#   #   				T = x_axis_info[2]
#   #   				cts = xx
#   #   				xx = array(cts) / T
#   #   				if max(xx) > .4:
#   #   					xstep = .1
#   #   				elif max(xx) > .15:
#   #   					xstep = .05
#   #   				elif max(xx) > .07:
#   #   					xstep = .02
#   #   				else:
#   #   					xstep = .01
#   #   				xmax = min(ceil(max(xx)*1.1/xstep)*xstep,1)
#   #   				xmin = 0
#   #   				ax.set_xlim( xmin, xmax )
#   #   				x_ticks = arange(xmin,xmax+xstep/2,xstep)
#   #   				ax.set_xticks( x_ticks )
#   #   				ax.set_xticklabels( [str(int(x*100+.01))+'%' for x in x_ticks],
#   #   							fontproperties=font12)
#   #   				ax.set_xlabel(x_axis_info[1], fontproperties=font12)
#   #   			elif x_axis_info[0] == 'defined':
#   #   				dummy,xlabel,xmin,xmax,xstep,x_labels = x_axis_info
#   #   				x_ticks = arange(xmin,xmax+xstep/2,xstep)
#   #   				ax.set_xticks( x_ticks )
#   #   				ax.set_xticklabels( x_labels, fontproperties=font12)
#   #   				ax.set_xlabel(xlabel, fontproperties=font12)
#   #   			y_interbar_buffer,y_edge_buffer,bar_width = [GID[q_idx][param]
#   #   					for param in ('y_interbar_buffer','y_edge_buffer','bar_width')]
#   #   			ymin = 0
#   #   			ymax = 2*y_edge_buffer + N_yy*bar_width + (N_yy-1)*y_interbar_buffer
#   #   			ax.set_ylim( ymin, ymax )
#   #   			y_ticks = y_edge_buffer + bar_width/2 + arange(
#   #   									N_yy ) * (bar_width+y_interbar_buffer)
#   #   			ax.set_yticks( y_ticks )
#   #   			ax.set_yticklabels( y_labels,fontproperties=font12, position=(-.01,0) )
#   #   			ax.tick_params(axis='both',left='off',right='off',top='off',bottom='off')
#   #   			#ax.set_title('('+str(q_idx)+')  '+GID[q_idx]['q_name'], 
#   #   			#	fontproperties=font14, loc='left')
#   #   			ax.set_title(GID[q_idx]['q_name'], 
#   #   				fontproperties=font14, loc='left')
#   #   			##################
#   #   			# bckgrnd & bars #
#   #   			for spine in ax.spines.values():
#   #   				spine.set_visible(False)
#   #   			bckgrnd_bar_midgap = xmax/100
#   #   			bckgrnd_bar_edgegap = 0
#   #   			bb_left_edges = x_ticks[:-1] + bckgrnd_bar_midgap/2
#   #   			bb_widths = [xstep - bckgrnd_bar_midgap] * len(bb_left_edges)
#   #   			bb_left_edges[0] = xmin + bckgrnd_bar_edgegap
#   #   			bb_widths[0] = bb_left_edges[1] - bb_left_edges[0] - bckgrnd_bar_midgap
#   #   			bb_widths[-1] = xmax - bb_left_edges[-1] - bckgrnd_bar_edgegap
#   #   			ax.bar( bottom=ymin,  height=[ymax-ymin]*len(x_ticks[1:]),
#   #   					left=bb_left_edges, width=bb_widths, 
#   #   					color=(0,0,0),alpha=.03,edgecolor=(1,1,1))
#   #   			if type(barcolor_input)==list:
#   #   				barcolor = barcolor_input
#   #   			elif type(barcolor_input)==dict:
#   #   				if q_idx in barcolor_input:
#   #   					barcolor = barcolor_input[q_idx]
#   #   				else:
#   #   					barcolor = barcolor_input[0]
#   #   			else:
#   #   				barcolor = [.4,.2,.9]
#   #   			ax.bar( left=0, height=bar_width, width=xx, bottom=y_ticks-bar_width/2,
#   #   					color=barcolor,alpha=.8,orientation='horizontal',
#   #   					edgecolor=barcolor)
#   #   			if 'print xx' in x_axis_info:
#   #   				for ct,x,y in zip(cts,xx,y_ticks):
#   #   					ax.text(x+xmax/100,y,'('+str(ct)+')',color=(.5,.5,.5),
#   #   							alpha=.6,ha='left',va='center',fontproperties=font12)
#   #   			elif 'print pp' in x_axis_info:
#   #   				for ct,x,y in zip(cts,xx,y_ticks):
#   #   					ax.text(x+xmax/100,y,'('+str(round(x*100,1))[:4]+'%)',color=(.5,.5,.5),
#   #   							alpha=.6,ha='left',va='center',fontproperties=font12)
#   #   			elif 'print x&p' in x_axis_info:
#   #   				for ct,x,y in zip(cts,xx,y_ticks):
#   #   					ax.text(x+xmax/100,y,
#   #   							'('+str(ct)+': '+str(round(x*100,1))[:4]+'%)',
#   #   							color=(.5,.5,.5),
#   #   							alpha=.6,ha='left',va='center',fontproperties=font12)
#   #   			##################
#   #   			#   ##########   #
#   #   		fig.savefig(pp, format='pdf')
#   #   		if close_figures:
#   #   			plt.close(fig)
#   #   	pp.close()



# previous draft, number 2:
#   #   def generate_bar_graph_pages0(
#   #   	graph_data0,
#   #   	barcolor_input=[.4,.7,.3],
#   #   	pg_len_limit=None,
#   #   	filename='test_graphs',
#   #   	close_figures=True,
#   #   	graph_param_input={},
#   #   	figtitlefont = font18,
#   #   	extra_label_space = 2.5,
#   #   	bar_to_buffer_ratio = .7,
#   #   												):
#   #   	char_width1 = { # helvetica size 12
#   #   		"'":                        0.1303552,
#   #   		'ijl':                      0.1515136,
#   #   		' ftI./!,:;':               0.189696,
#   #   		'r-()':                     0.227392,
#   #   		'"':                        0.2422272,
#   #   		'*':                        0.2655744,
#   #   		'^':                        0.2718976,
#   #   		'cksvxyzJ':                 0.3409664,
#   #   		'abdeghnopquL0123456789#$': 0.3793920,
#   #   		'=':                        0.3983616,
#   #   		'FTZ':                      0.4168448,
#   #   		'ABEKPSVXY&':               0.4552704,
#   #   		'wCDHNRU':                  0.4927232,
#   #   		'GOQ':                      0.5309056,
#   #   		'mM':                       0.5683584,
#   #   		'%':                        0.6072704,
#   #   		'W':                        0.6444800,
#   #   		'@':                        0.6914176,   }
#   #   	char_width2 = {c:x for cc,x in char_width1.items() for c in cc}
#   #   	char_width2.update({'default':0.4})
#   #   	graph_params = {
#   #   		'left_margin': {0:.2},
#   #   		'header_buffer': {0:1.5},
#   #   		'super_buffer': {0:1.5},
#   #   		'sub_buffer': {0:2.2},
#   #   		'y_edge_buffer': {0:.3},
#   #   		'y_interbar_buffer': {0:.2},
#   #   		'bar_width': {0:.8},
#   #   									}
#   #   	graph_params.update({
#   #   		param: param_val if type(param_val)==dict else {0:param_val}
#   #   							for param,param_val in graph_param_input.items()
#   #   							if param in graph_params})
#   #   	# graph_info_dict: this is where we will be putting most of our info
#   #   	GID = { q_idx+1: 
#   #   		{   'q_name':q_data[0],
#   #   			'x_axis_info':q_data[1],
#   #   			'xy_data':q_data[2],
#   #   			'group_idx':group_idx,
#   #   			'N_yy':len(q_data[-1])   } # = len(q_data[2])
#   #   				for q_idx,(q_data,group_idx) in enumerate(
#   #   				[  (q_data,group_idx) 
#   #   				for group_idx,(group_name,group_data) in enumerate(graph_data0)
#   #   				for q_data in group_data] ) }
#   #   	for q_idx,graph_info in GID.items():
#   #   		for param,param_val in graph_params.items():
#   #   			if q_idx in param_val:
#   #   				graph_info[param] = param_val[q_idx]
#   #   			else:
#   #   				graph_info[param] = param_val[0]
#   #   		# now we figure out how much space is needed between labels:
#   #   		lb_cts = [lab.count('\n') if type(lab)==str else 0
#   #   						for lab,x in GID[q_idx]['xy_data']]
#   #   		lb_tups = {(l1,l2) for l1,l2 in zip(lb_cts[:-1],lb_cts[1:])}
#   #   		#### i think the following parameters are good enough to keep,
#   #   		#### i.e., no customization options necessary.  
#   #   		# customizable kwargs:
#   #   		# - extra_label_space: space between labels.  2 ~ double spacing.
#   #   		# - bar_to_buffer_ratio: when labels get further apart, the expansion
#   #   		#   is shared in the graph between the bars and the space between the bars.
#   #   		#   the b_to_b ratio determines how much of this expansion is given to the 
#   #   		#   bars.  if it is 1, then the interbar buffer is constant.  
#   #   		char_hgt = .479 # height of a helvetica pt12 character line
#   #   		spc_hgt = .234  # height of a helvetica pt12 space between lines
#   #   		lab_spacing = max( (lb1+lb2)/2 * spc_hgt          # label linebreaks
#   #   							+ (lb1+lb2+2)/2 * char_hgt    # label line (chars)
#   #   							+ spc_hgt * extra_label_space # space btwn labels
#   #   							for lb1,lb2 in lb_tups)
#   #   		bar_width = bar_to_buffer_ratio * lab_spacing
#   #   		graph_info['bar_width'] = bar_width
#   #   		graph_info['y_interbar_buffer'] = lab_spacing - bar_width
#   #   		graph_info['subplot_hgt'] = (
#   #   				(graph_info['N_yy']-1) 
#   #   				   * graph_info['y_interbar_buffer']
#   #   				+ 2 * graph_info['y_edge_buffer']
#   #   				+ graph_info['N_yy']
#   #   				   * graph_info['bar_width']	)
#   #   		graph_info['super_buffer'] = graph_info['q_name'].count('\n')
#   #   	########################################
#   #   	# here we distribute our graphs to pages
#   #   	graph_page_data = {} # {page_idx: [figure_title,[q_indices],figure_height]}
#   #   	prev_group_idx = -1 # dummy -1, group_idx starts at 0
#   #   	q_idx = 1; page_idx = 0 # note that q_idx and page_idx BOTH start at 1
#   #   	while q_idx in GID:
#   #   		group_idx = GID[q_idx]['group_idx']
#   #   		add_hgt = sum(GID[q_idx][param] for param in 
#   #   							('subplot_hgt','super_buffer','sub_buffer'))
#   #   		if group_idx != prev_group_idx:
#   #   			page_idx += 1
#   #   			group_name = graph_data0[group_idx][0]
#   #   			graph_page_data[page_idx] = [group_name,[q_idx],
#   #   					sum(GID[q_idx][param] for param in 
#   #   							('subplot_hgt','super_buffer','sub_buffer',
#   #   								'header_buffer'))] # header buffer probably constant
#   #   		elif (pg_len_limit==None or 
#   #   				graph_page_data[page_idx][2] + add_hgt <= pg_len_limit):
#   #   			graph_page_data[page_idx][1].append(q_idx)
#   #   			graph_page_data[page_idx][2] += add_hgt
#   #   		else:
#   #   			page_idx += 1
#   #   			if group_name!=None:
#   #   				group_name = group_name + ' (continued)'
#   #   			graph_page_data[page_idx] = [group_name,[q_idx],
#   #   					sum(GID[q_idx][param] for param in 
#   #   						('subplot_hgt','super_buffer','sub_buffer','header_buffer'))]
#   #   		prev_group_idx = group_idx
#   #   		q_idx += 1
#   #   	pp = PdfPages(filename+'.pdf')
#   #   	for page_idx,(fig_title,q_indices,fig_hgt) in sorted(graph_page_data.items()):
#   #   		N_subplots = len(q_indices)
#   #   		hratios = [GID[q_idx]['subplot_hgt'] for q_idx in q_indices]
#   #   		avg_subplot_hgt = sum(hratios)/N_subplots
#   #   		""" note that only one value is taken for the header/sub-/sup-buffers 
#   #   			on a single page.  the first q_idx is used:                       """
#   #   		header_buffer,super_buffer,sub_buffer = [ GID[min(q_indices)][param]
#   #   					for param in ('header_buffer','super_buffer','sub_buffer')]
#   #   		top_margin = 1 - (header_buffer + super_buffer) / fig_hgt
#   #   		bottom_margin = sub_buffer / fig_hgt
#   #   		height_space = (super_buffer + sub_buffer) / avg_subplot_hgt
#   #   		right_margin = .96 # should be fine for all cases; if not make this a param
#   #   		fig = plt.figure(figsize=(8,fig_hgt/4))
#   #   		fig.suptitle(fig_title+'  ('+str(page_idx)+')',x=.03,y=1-.03*20/fig_hgt,
#   #   								fontproperties=figtitlefont,ha='left')
#   #   		axx = []
#   #   		for q_idx in sorted(q_indices):
#   #   			x_axis_info = GID[q_idx]['x_axis_info']
#   #   			y_labels,xx = zip(*GID[q_idx]['xy_data'][::-1])
#   #   			N_yy = GID[q_idx]['N_yy'] #  = len(y_labels)
#   #   			""" note that left_margin is the sole difference in gridspecs of
#   #   			    subplots on the same page (i.e., in the same figure)         """
#   #   			#if text_space_override==False:
#   #   			#	left_margin = GID[q_idx]['left_margin']
#   #   			#else:
#   #   			max_lab_len = max([sum(char_width2[c] if c in char_width2
#   #   							else char_width2['default'] for c in sublab)
#   #   							for lab in y_labels for sublab in str(lab).split('\n')])
#   #   			left_margin = max_lab_len / 32 + .03
#   #   
#   #   
#   #   			gs = gridspec.GridSpec(N_subplots, 1,
#   #   				bottom=bottom_margin, top=top_margin,
#   #   				hspace=height_space, height_ratios=hratios,
#   #   				left=left_margin, right=right_margin )
#   #   			ax = fig.add_subplot( gs[q_idx-min(q_indices)] ); axx.append(ax)
#   #   			##################
#   #   			# ticks & labels #
#   #   			if x_axis_info[0] == 'percentage':
#   #   				""" For percs, x_axis_info should be ['percentage', XLABEL,
#   #   				    TOTAL #, 'print xx' (or not - this prints the ct)]       """
#   #   				T = x_axis_info[2]
#   #   				cts = xx
#   #   				xx = array(cts) / T
#   #   				if max(xx) > .4:
#   #   					xstep = .1
#   #   				elif max(xx) > .15:
#   #   					xstep = .05
#   #   				elif max(xx) > .07:
#   #   					xstep = .02
#   #   				else:
#   #   					xstep = .01
#   #   				xmax = min(ceil(max(xx)*1.1/xstep)*xstep,1)
#   #   				xmin = 0
#   #   				ax.set_xlim( xmin, xmax )
#   #   				x_ticks = arange(xmin,xmax+xstep/2,xstep)
#   #   				ax.set_xticks( x_ticks )
#   #   				ax.set_xticklabels( [str(int(x*100+.01))+'%' for x in x_ticks],
#   #   							fontproperties=font12)
#   #   				ax.set_xlabel(x_axis_info[1], fontproperties=font12)
#   #   			elif x_axis_info[0] == 'defined':
#   #   				dummy,xlabel,xmin,xmax,xstep,x_labels = x_axis_info
#   #   				x_ticks = arange(xmin,xmax+xstep/2,xstep)
#   #   				ax.set_xticks( x_ticks )
#   #   				ax.set_xticklabels( x_labels, fontproperties=font12)
#   #   				ax.set_xlabel(xlabel, fontproperties=font12)
#   #   			y_interbar_buffer,y_edge_buffer,bar_width = [GID[q_idx][param]
#   #   					for param in ('y_interbar_buffer','y_edge_buffer','bar_width')]
#   #   			ymin = 0
#   #   			ymax = 2*y_edge_buffer + N_yy*bar_width + (N_yy-1)*y_interbar_buffer
#   #   			ax.set_ylim( ymin, ymax )
#   #   			y_ticks = y_edge_buffer + bar_width/2 + arange(
#   #   									N_yy ) * (bar_width+y_interbar_buffer)
#   #   			ax.set_yticks( y_ticks )
#   #   			ax.set_yticklabels( y_labels,fontproperties=font12, position=(-.01,0) )
#   #   			ax.tick_params(axis='both',left='off',right='off',top='off',bottom='off')
#   #   			#ax.set_title('('+str(q_idx)+')  '+GID[q_idx]['q_name'], 
#   #   			#	fontproperties=font14, loc='left')
#   #   			ax.set_title(GID[q_idx]['q_name'], 
#   #   				fontproperties=font14, loc='left')
#   #   			##################
#   #   			# bckgrnd & bars #
#   #   			for spine in ax.spines.values():
#   #   				spine.set_visible(False)
#   #   			bckgrnd_bar_midgap = xmax/100
#   #   			bckgrnd_bar_edgegap = 0
#   #   			bb_left_edges = x_ticks[:-1] + bckgrnd_bar_midgap/2
#   #   			bb_widths = [xstep - bckgrnd_bar_midgap] * len(bb_left_edges)
#   #   			bb_left_edges[0] = xmin + bckgrnd_bar_edgegap
#   #   			bb_widths[0] = bb_left_edges[1] - bb_left_edges[0] - bckgrnd_bar_midgap
#   #   			bb_widths[-1] = xmax - bb_left_edges[-1] - bckgrnd_bar_edgegap
#   #   			ax.bar( bottom=ymin,  height=[ymax-ymin]*len(x_ticks[1:]),
#   #   					left=bb_left_edges, width=bb_widths, 
#   #   					color=(0,0,0),alpha=.03,edgecolor=(1,1,1))
#   #   			if type(barcolor_input)==list:
#   #   				barcolor = barcolor_input
#   #   			elif type(barcolor_input)==dict:
#   #   				if q_idx in barcolor_input:
#   #   					barcolor = barcolor_input[q_idx]
#   #   				else:
#   #   					barcolor = barcolor_input[0]
#   #   			else:
#   #   				barcolor = [.4,.2,.9]
#   #   			ax.bar( left=0, height=bar_width, width=xx, bottom=y_ticks-bar_width/2,
#   #   					color=barcolor,alpha=.8,orientation='horizontal',
#   #   					edgecolor=barcolor)
#   #   			if 'print xx' in x_axis_info:
#   #   				for ct,x,y in zip(cts,xx,y_ticks):
#   #   					ax.text(x+xmax/100,y,'('+str(ct)+')',color=(.5,.5,.5),
#   #   							alpha=.6,ha='left',va='center',fontproperties=font12)
#   #   			elif 'print pp' in x_axis_info:
#   #   				for ct,x,y in zip(cts,xx,y_ticks):
#   #   					ax.text(x+xmax/100,y,'('+str(round(x*100,1))[:4]+'%)',color=(.5,.5,.5),
#   #   							alpha=.6,ha='left',va='center',fontproperties=font12)
#   #   			elif 'print x&p' in x_axis_info:
#   #   				for ct,x,y in zip(cts,xx,y_ticks):
#   #   					ax.text(x+xmax/100,y,
#   #   							'('+str(ct)+': '+str(round(x*100,1))[:4]+'%)',
#   #   							color=(.5,.5,.5),
#   #   							alpha=.6,ha='left',va='center',fontproperties=font10)
#   #   			##################
#   #   			#   ##########   #
#   #   		fig.savefig(pp, format='pdf')
#   #   		if close_figures:
#   #   			plt.close(fig)
#   #   	pp.close()










