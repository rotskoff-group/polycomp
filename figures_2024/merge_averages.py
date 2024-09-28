#Script for merging average values

import cupy as cp
import sys
import math

cp.random.seed(int(sys.argv[1]))

rand_list = cp.sort(cp.random.rand(5))
norm_amt = {
        'ABC' : rand_list[0],
        'ABCCBA' : rand_list[1] - rand_list[0],
        'CAABBC' : rand_list[2] - rand_list[1],
        'AB' : rand_list[3] - rand_list[2],
        'AC' : rand_list[4] - rand_list[3],
        'BC' : 1 - rand_list[4],
        }

#tot = sum(norm_amt.values())
#for key in norm_amt.keys():
#    norm_amt[key] = norm_amt[key] / tot
if not math.isclose(sum(norm_amt.values()), 1.0):
    raise ValueError("Norm dict not sum to 1")
print(norm_amt)

array = cp.array([[int(sys.argv[1]), norm_amt['ABC'].get(), norm_amt['ABCCBA'].get(), norm_amt['CAABBC'].get(), norm_amt['AB'].get(), norm_amt['AC'].get(), norm_amt['BC'].get(), 0]]) 

old_array = cp.load('/scratch/users/epert/ABC_neural/production_ABC/plotting_scripts/merged_array.npy')

new_array = cp.concatenate((old_array, array))
print(new_array.shape)
cp.save('merged_array.npy', new_array)



exit()

array = cp.array([[int(sys.argv[1]), norm_amt['ABC'].get(), norm_amt['ABCCBA'].get(), norm_amt['CAABBC'].get(), norm_amt['AB'].get(), norm_amt['AC'].get(), norm_amt['BC'].get(), 0]]) 
print(array.shape)

cp.save('merged_array.npy', array)
print(array.get())
