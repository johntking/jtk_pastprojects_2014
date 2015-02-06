'''
I wrote this to teach someone how to use list comprehensions.  Pretty basic, but important to master
for data analysis (in my opinion).  

'''


# general form: [EXPRESSEION for ELEMENT in LIST if CONDITION]

###################
list1 = [1,2,3,4,5]
###################

# basic list comp: returns the same list:
[x for x in list1]

# any term after "for" is a variable whose name should be thought
# of as existing only within the list comprehension. we could 
# name it whatever we want, it doesn't matter. however, we CANNOT
# put an expression as this term (just like you can't write, on 
# a separate line, "x+1 = 3"):
[y for y in list1]
[gagagegoo for gagagegoo in list1]

# the first term can be an expression (of x or otherwise):
[(x-1)**2 for x in list1]

# the expression doesn't have to be of x:
[1 for x in list1]

# we can add a condition (here: that x is not divisible by two):
[(x-1)**2 for x in list1 if x%2==1]

# terms preceded by "in" can also be modified -- 
# any iterable object can be put in this term:
[(x-1)**2 for x in list1[:3]]
[x for x in list1+list1+list1]
[x+1 for x in [y for y in ]]

# note the difference in order of operation between these two LC's:
[(x-1)**2 for x in list1[:3] if x%2==1]
[(x-1)**2 for x in list1 if x%2==1][:3]

##################
list2 = [10,20,30]
##################

# iterate over two lists in one list comprension? no problem.
# the following is the cross product of the two lists:
[(x,y) for x in list1 for y in list2]

# the same could be done with a single list:
[(x,y) for x in list1 for y in list1]

# we can also put a mathematical expression in, just as before:
[x+y for x in list1 for y in list2]

# note that the following two LC's are equivalent; the
# variable names are swapped, but there is no real difference:
[x**2+y for x in list1 for y in list2]
[y**2+x for y in list1 for x in list2]

# conditions can be added as before:
[x+y for x in list1 for y in list2 if y%x==0]

######################################
list3 = [[1,2,3],[4,5,6],[7,8,9],[10]]
######################################

# basic way to flatten a list:
[x for sublist in list3 for x in sublist]

# remember that the terms after "in" can be (iterable) expressions:
[x for sublist in list3[1:] for x in sublist[:2]]
[x**2 for sublist in list3[1:3] for x in sublist[1:]]

# and we can get more complicated...
[x+y for sublist in list3 for x in sublist for y in list1]

# the order can be changed: all we require is the the final
# expression comes first ("x+y"), that each for-in pair is
# kept together ("for y in list1" can be moved around, as
# long as "for y" and "in list1" are not separated), and 
# that any term with a variable introduced by another must
# follow the term where it is introduced (so "for sublist in
# list" must precede "for x in sublist").  any condition,
# if present, always comes LAST.  
[x+y for sublist in list3 for y in list1 for x in sublist if x**3>y]

# we can unpack list, just as in a for-clause; make sure each
# item we are unpacking has the appropriate number of sub-items!
# here, if we didn't slice list3, we would raise an error by trying
# to unpack three items from [10] (when there is only one available):
[x*z for x,y,z in list3[:3]]

# note that we can make dict and set comprehensions, as well as generators:
{x%4 for x in list1}
{sum(y):y for y in list3}
(z%7 for z in list2)

################################################
dict1 = {x: (x*10 + x%7)%19 for x in range(2,9)}
################################################

# we can make list (or dict, set...) comprehensions from this,
# using dict methods .keys(), .values() and items():
[x for x in dict1.keys() if x > 2 and x % 2 == 1]

# the above is the same as this (here you can omit ".keys()"):
[x for x in dict1 if x > 2 and x % 2 == 1]

# note that
{x for x in dict1} == set(dict1.keys()) == set(dict1)

# and now let's throw in a little set logic:
set(dict1) & {y%10 for y in dict1.values()} & {x//4 for x in dict1.values()}

# we can use dict comps to reverse a dictionary, but
# this only works (cleanly) if the dict values are unique:
{value:key for key,value in dict1.items()}

# weird stuff:
{dict1[x] for x in dict1.values() if x in dict1.keys()}

# and i'll finish with a favorite list comprehension technique
# i use all the time. if i have a dict, or iterable of iterables
# (e.g., a list of 3-tuples), i often want to make an expression 
# of one of the sub-objects under a condition of another:

[y**3//(y+1) for x,y in dict1.items() if x % 3 != 1]





