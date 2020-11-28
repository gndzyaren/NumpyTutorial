# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:35:32 2020

@author: yg
"""
print("#***********************#CREATING ARRAYS AND DATA TYPES#*************************#")

                                #CREATING ARRAYS AND DATA TYPES
import numpy as np
import matplotlib.pylab as plt
from numpy import random

list_1 = np.array([1,2,3,4,5])
print(list_1)

np_arr_1 = np.array(list_1,dtype=np.int8) #giving the numpy arrat data type as byte
print(np_arr_1)

#Multidimensional array creation
m_list_1 = [[1,2,3],[4,5,6],[7,8,9]]
np_m_arr_1 =np.array(m_list_1)
print(np_m_arr_1)

arange_array = np.arange(1,10,2) #Writing the array from the 1 to the 10 increased by 2
print(arange_array)

linspace_array = np.linspace(0,5,7) #7 elements on array within the 0 lower bound and 5 upper bound
print(linspace_array)
print(np.zeros(4)) # One raw 4 column matrix

zeros_array = np.zeros((2,3))
print(zeros_array)

print(np_m_arr_1.size) #returns the elements of the array
np_arr_2 = np.array([1,2,3,4,5,6])
print(np_arr_2.dtype)


rand_array = np.random.randint(10,50,size=(2,3)) #returns the 2*3 matrix from the bounded by 10 and 50
print(rand_array)
#print(np.random.randint?) #returns the helping information about random


print("#***********************#SLICING AND INDEXING#*************************#")
                                #SLICING AND INDEXING
np_m_arr_1[0,0] = 2 #updates the first column first row value as 2
np_m_arr_1.itemset((0,1),1) #updates the first row,second column as 1
np_m_arr_1.shape #return the dimensions of the array as tuple
np_m_arr_1[0,1] #gives the first row,second column value
np_m_arr_1.item(0,1) #selects the first row,second column value

print(np_m_arr_1)

np.take(np_m_arr_1,[0,3,6]) #take the 0th,3rd,6th index
np.put(np_m_arr_1,[0, 3 ,6],[10 ,10, 10])
print(np_m_arr_1)
#np_arr_1 1 2 3 4 5
print(np_arr_1)
print(np_arr_1[:5:2])
print(np_arr_1[::1])
print(np_arr_1[::-1])

evens = np_m_arr_1[np_m_arr_1 %2 ==0]
evens
print(np_m_arr_1[np_m_arr_1 > 5])
print(np_m_arr_1[(np_m_arr_1 > 5) | (np_m_arr_1 < 9)])
print(np_m_arr_1[(np_m_arr_1 > 5) & (np_m_arr_1 == 10)])
print(np.unique(np_m_arr_1))

print("#**********************#RESHAPING ARRAYS#**************************#")
                                #RESHAPING ARRAYS
print(np_m_arr_1.reshape((1,9))) #1*9 matrix
print(np.resize(np_m_arr_1,(2,5))) #2*5 matrix
print(np_m_arr_1)
print(np_m_arr_1.transpose())
print(np_m_arr_1.swapaxes(0,1))
print(np_m_arr_1.flatten('F')) #writing columns as rows
print(np_m_arr_1.sort(axis=0))# sorting as rows
print(np_m_arr_1)

print("#**********************#STACKING AND SPLITTING#**************************#")
                                 #STACKING AND SPLITTING

ss_arr_1 = np.random.randint(10,size=(2,2))
print(ss_arr_1)

ss_arr_2 = np.random.randint(10,size=(2,2))
print(ss_arr_2)

print(np.vstack((ss_arr_1,ss_arr_2)))
print(np.hstack((ss_arr_1,ss_arr_2)))


ss_arr_3 = np.delete(ss_arr_1,1,0)
print(ss_arr_3)

ss_arr_4 = np.delete(ss_arr_2,1,0)
print(ss_arr_4)
col_stack = np.column_stack((ss_arr_3,ss_arr_4))
print(col_stack)

ss_arr_5 = np.random.randint(10,size=(2,10))
print(ss_arr_5)

print(np.hsplit(ss_arr_5,5)) #splitting to 5 sub-array


print("#**********************#COPYING#**************************#")
                                #COPYING
cp_arr_1 = np.random.randint(10,size=(2,2))
cp_arr_2 = cp_arr_1
print(cp_arr_1)
print(cp_arr_2)

cp_arr_2 = cp_arr_1.copy()
print(cp_arr_2)
print(cp_arr_1)

cp_arr_3 = cp_arr_1.view()
cp_arr_3.flatten('F')
print(cp_arr_1)
print(cp_arr_3)
cp_arr_4 = cp_arr_1.copy()

print("#**********************#BASIC MATHEMATICS#**************************#")
                                #BASIC MATHEMATICS
arr_3 = np.array([1,2,3,4])
arr_4 = np.array([2,4,6,8])
arr_3 / arr_4
arr_5 = random.randint(100,size=(4))
arr_6 = random.randint(100,size=(2,3))
print(arr_6)
print(arr_6.sum(axis=0)) #sums the all columns
print(arr_6.cumsum(axis=1)) #sums the all rows as cumulative way

np.add(arr_3,5)
#np.sub(arr_3,arr_4)
np.divide(arr_3,arr_4)
np.multiply(arr_3,arr_4)
arr_5 = np.array([[1,2],[3,4]])
arr_6 = np.array([[2,4],[6,9]])
np.remainder(arr_6,arr_5)
np.power(arr_6,arr_5)
np.absolute(arr_5,arr_6)
np.exp(arr_5)
np.log(arr_3)
np.log2(arr_5)
np.log10(arr_6)

print(np.gcd.reduce([9,12,15])) #ebob
print(np.lcm.reduce([9,12,15])) #ekok

np.floor([1.2 , 3.5])
np.ceil([1.2 , 3.5])

sq_arr = np.arange(6)**2 #returns the square array up to 6 
print(sq_arr)

arr_7 = np.random.randint(10,size=(5,3))
mc_index = arr_7.argmax(axis=0)
max_nums = arr_7[mc_index]
print(max_nums)

print("#**********************#READING FROM FILES#**************************#")
                                #READING FROM FILES
import pandas as pd
from numpy import genfromtxt

ic_sales = pd.read_csv(r"C:\\Users\\yaren\\Desktop\\NumpyCalismalarim\\IceCreamData.csv").to_numpy()
print(ic_sales)

ic_sales_2 = genfromtxt(r"C:\\Users\\yaren\\Desktop\\NumpyCalismalarim\\IceCreamData.csv",delimiter=',')
ic_sales_2 = [ row[~np.isnan(row)] for row in ic_sales_2]
print(ic_sales_2)

print("#**********************#STATISTICS FUNCTIONS#**************************#")
                                    #STATISTICS FUNCTIONS

sarr_1 = np.arange(6)
sarr_1
np.mean(sarr_1)
np.median(sarr_1)
np.std(sarr_1)
np.var(sarr_1)
np.average(sarr_1)
print(ic_sales)
print(np.percentile(ic_sales,50,axis=0))
print(ic_sales[:,0])
print(ic_sales[:,1])

print(np.corrcoef(ic_sales[:,0],ic_sales[:,1]))

#Calculate regression lines
temp_mean = np.mean(ic_sales[:,0])
sales_mean = np.mean(ic_sales[:,1])
numerator = np.sum(((ic_sales[:,0] - temp_mean) *(ic_sales[:,1] - sales_mean)))
determinator = np.sum(np.square(ic_sales[:,0] - temp_mean))
slope = numerator / determinator
y_i = sales_mean - slope * temp_mean
print(y_i)
reg_arr = ic_sales[:,0] * slope + y_i
print(reg_arr)


print("#**********************#TRIGONOMETRY FUNCTIONS#**************************#")
                                    #TRIGONOMETRY FUNCTIONS
t_arr = np.linspace(np.pi,np.pi,200)
plt.plot(t_arr,np.sin(t_arr))
print(np.arcsin(1))
print(np.arccos(1))
np.arctan(1)
np.rad2deg(np.pi)
np.deg2rad(180)
np.hypot(10,10)

print("#**********************#LINEAR ALGEBRA FUNCTIONS#**************************#")
                                    #LINEAR ALGEBRA FUNCTIONS

from numpy import linalg as LA
print("arr_5\n",arr_5)
print("arr_6\n",arr_6)

arr_8 = np.array([[5,6],[7,8]])


np.inner(arr_5,arr_6)
arr_9 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
arr_10 = np.array([[1,2],[3,4]],dtype=object)
np.tensordot(arr_9,arr_10)


arr_11 = np.array([0,1])
arr_12 = np.array([[1,2,3,4],[5,6,7,8]])
np.einsum('i,ij->i',arr_11,arr_12)
LA.matrix_power(arr_5,2)
np.kron(arr_5,arr_6)
LA.eig(arr_5)
LA.eig(arr_5)
LA.norm(arr_5)

LA.inv(arr_5)
LA.cond(arr_5)

arr_12 = np.array([[1,2],[3,4]])
LA.det(arr_12)

arr_12_i = LA.inv(arr_12)
#LA.dot(arr_12,arr_12_i)

arr_13 = np.array([[1,4],[6,18]])
arr_14 = np.array([10,42])
LA.solve(arr_13,arr_14)
np.eye(2,2,dtype=int)

print("#**********************#SAVING AND LOADING NUMPY OBJECTS#**************************#")
                                    #SAVING AND LOADING NUMPY OBJECTS

arr_15 = np.array([[1,2],[3,4]])
np.save('randarray',arr_15)
arr_16 = np.load('randarray.npy')
arr_16
np.savetxt('randcsv.csv',arr_15)
arr_17 = np.loadtxt('randcsv.csv')
print(arr_17)


print("#**********************#FINANCIAL FUNCTIONS#**************************#")
                                    #FINANCIAL FUNCTIONS
import numpy_financial as npf
npf.fv(8/12,10*12,-400,400)
period = np.arange(1 * 12) + 1
principle = 3000.00
ipmt = npf.ipmt(0.0925/12,period,1*12,principle)
ppmt = npf.ipmt(0.0925/12,period,1*12,principle)

for payment in period:
    index = payment -1
    principle = principle + ppmt[index]
print(f"{payment} {np.round(ppmt[index],2)} {np.round(ipmt[index],2)} {np.round(principle,2)}")

np.round(npf.nper(0.0925/12,-150,3000.00),2)
npf.npv(0.08,[-1500,4000,5000,6000,7000])


print("#**********************#COMPARISON FUNCTIONS#**************************#")
                                    #COMPARISON FUNCTIONS
carr_1 = np.array([2,3])
carr_2 = np.array([3,2])
np.greater(carr_1,carr_2)
np.greater_equal(carr_1,carr_2)
np.less(carr_1,carr_2)
np.less_equal(carr_1,carr_2)
np.not_equal(carr_1,carr_2)
np.equal(carr_1,carr_2)