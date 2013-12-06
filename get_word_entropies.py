#! /usr/bin/python

import sys
import math
from os import system
import numpy

count_file=sys.argv[1] #ex: temp.word_count
unselected_words_file=count_file+'.unselected'

counts=numpy.genfromtxt(count_file,dtype='float')
cumulative_counts=numpy.sum(counts, axis=0)
normalized_count=numpy.divide(counts,cumulative_counts)

probability_normalizer=numpy.sum(normalized_count, axis=1)

probability_normalizer=numpy.tile(probability_normalizer,(counts.shape[1],1))  
probability_normalizer=probability_normalizer.T
probabilities=numpy.divide(normalized_count,probability_normalizer)
probabilities=numpy.where(probabilities==0.0,0.0000000001,probabilities)

log_probabilities=numpy.log(probabilities)
entropies = -1*numpy.sum(numpy.multiply(log_probabilities,probabilities),axis=1)

# unselect words based on low count and high entropy
probability_normalizer=numpy.sum(normalized_count, axis=1) # this is also the normalized word count, normalized by the class frquency
min_word_NC = numpy.min(probability_normalizer)
ent_percentile = numpy.percentile(numpy.array(entropies.T), 99, axis=0)
print min_word_NC

unselected_words = numpy.logical_or((probability_normalizer < 2*min_word_NC),(entropies > ent_percentile ))
#unselected_words = (probability_normalizer < 2*min_word_NC)
#unselected_words = (entropies > ent_percentile )

numpy.savetxt(unselected_words_file,unselected_words)


