#! /usr/bin/python

import sys
import math
from os import system
import numpy

# get the directory 
dir_current=sys.argv[1] # give the name of the directory where all the files are stored

# concatenate results of all files together
command = 'cat '+dir_current+'/*/*prediction_n_true | sed \'s/couns\.ques/0/g;s/couns\.gi/1/g;s/couns\.fa/2/g;s/couns\.ref/3/g;s/couns\.other/4/g\' > all_results'
system(command)

results=numpy.genfromtxt('all_results',dtype='int')

# calculate accuracy
true_values = (results[:,0]==results[:,1])
accuracy = numpy.sum(true_values)/float(results.shape[0])
print accuracy

num_classes = numpy.unique(results[:,0]).shape
confusion_mat=numpy.zeros((num_classes[0],num_classes[0]))
# calculate confusion matrix
for instance_id in range(results.shape[0]):
	confusion_mat[results[instance_id,0],results[instance_id,1]] += 1	

print confusion_mat

# get accuracies
instances_sum = numpy.sum(confusion_mat, axis=1)
instances_sum = numpy.tile(instances_sum,(num_classes[0],1))
instances_sum = instances_sum.T
norm_confusion_mat = numpy.divide(confusion_mat,instances_sum)
print norm_confusion_mat

# get unw accuracy
cum_acc_class=0
for class_id in range(num_classes[0]):
	cum_acc_class += norm_confusion_mat[class_id,class_id]
unw_acc = cum_acc_class/num_classes[0]

print unw_acc	
