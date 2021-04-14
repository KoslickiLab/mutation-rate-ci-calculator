#!/usr/bin/env python3
"""
Simulate pairs of sequences of values chosen from the unit interval under a
mutation model, counting the number of mutated kmers.

A 'sequence pair' is a random linear sequence and a mutated version of it. The
mutated version is consistent with a model where the sequence represents hash
values of kmers in a sequence of nucleotides and the individual nucleotides are
subject to error, with the caveat that no duplicated kmers are allowed.

Note that we don't actually generate sequence pairs. Instead, we generate the
positions of the mutations in the conceptual nucleotide sequence and derive the
relevant values from those.

Example 1:

  kmerSequenceLength = 100000
  kmerSize = 21
  pSubstitution = 0.05
  mutationModel = MutationModel(kmerSequenceLength+kmerSize-1,kmerSize,pSubstitution)
  for trialNumber in range(100):
    mutationModel.generate()
    (nErrors,nMutated) = mutationModel.count()
    print(nMutated)

Example 2:

  kmerSequenceLength = 100000
  kmerSize = 21
  pSubstitution = 0.05
  sketchSizes = [100,200]
  mutationModel = MutationModel(kmerSequenceLength+kmerSize-1,kmerSize,pSubstitution,
                                sketchSizes=sketchSizes)
  for trialNumber in range(100):
    mutationModel.generate()
    (nErrors,nMutated) = mutationModel.count()
    print(nMutated)
    for sketchSize in sketchSizes:
      nIntersection = mutationModel.simulate_sketch(nMutated,sketchSize)
      print(sketchSize,nIntersection)"""


from sys    import argv,stdin,stdout,stderr,exit
from random import Random
from hypergeometric_slicer import q_to_r1,q_to_jaccard,jaccard_to_r1,jaccard_to_q

try:
	from numpy.random import RandomState
	randomStateImported = True
except ModuleNotFoundError:
	randomStateImported = False

try:
	from scipy.stats import hypergeom
	hypergeomImported = True
except ModuleNotFoundError:
	hypergeomImported = False


# MutationModel--
#	Generate a sequence of poisson-type errors, and report the number of errors
#	and mutated kmers
#
# Implementation notes:
#	(1) We use separate PRNGs for errors and sketches so that we'll get the
#	same simulation results whether we do sketches or not (as long as the user
#	specifies a seed), and for a given sketch size, we'll get the same
#	simulation results regardless of what other sketch sizes we do.
#
#	(2) We use numpy.random random objects for sketches because of its support
#	for hypergeom.

class MutationModel(object):

	def __init__(self,ntSequenceLength,kmerSize,pSubstitution,prngSeed=None,sketchSizes=None,useNaiveCounter=False):
		if (kmerSize         < 1):        raise ValueError
		if (ntSequenceLength < kmerSize): raise ValueError
		if (not 0 <= pSubstitution <= 1): raise ValueError

		self.ntSequenceLength   = ntSequenceLength
		self.kmerSequenceLength = ntSequenceLength - (kmerSize-1)
		self.kmerSize           = kmerSize
		self.pSubstitution      = pSubstitution
		self.sketchSizes        = sketchSizes if (sketchSizes != None) else []
		self.mutatedKmerCounter = count_mutated_kmers_linear_naive if (useNaiveCounter) \
		                     else count_mutated_kmers_linear

		self.prng = Random()
		if (prngSeed != None):
			if (type(prngSeed) == str):
				self.prng.seed(prngSeed.encode("utf-8"))
			else:
				self.prng.seed(prngSeed)
		stopGapSeed = self.prng.randint(0,999999999)	# generate this even if we might not
														# .. need it, so that the prng will be
														# .. in the same state regardless

		if (sketchSizes != None):
			assert (hypergeomImported), \
			       "sketches aren't supported, because scipy.stats couldn't be imported"
			assert (randomStateImported), \
			       "sketches aren't supported, because numpy.random couldn't be imported"
			if (prngSeed == None):
				sketchSeedBase = stopGapSeed
			elif (type(prngSeed) == str):
				sketchSeedBase = prngSeed
			else:
				sketchSeedBase = str(prngSeed)
			self.sketchPrngs = {}
			for sketchSize in sketchSizes:
				sketchPrngSeed = "%s.%s" % (sketchSeedBase,sketchSize)
				self.sketchPrngs[sketchSize] = RandomState(seed=list(sketchPrngSeed.encode("utf-8")))

	def generate(self):
		self.errorSeq = list(map(lambda _:1 if (self.prng.random()<self.pSubstitution) else 0,range(self.ntSequenceLength)))
		return self.errorSeq

	def count(self,regenerate=False):
		if (regenerate): self.generate()
		return (sum(self.errorSeq),self.mutatedKmerCounter(self.kmerSize,self.errorSeq))

	def simulate_sketch(self,nMutated,sketchSize):
		if not (0 < sketchSize <= self.kmerSequenceLength): raise ValueError
		prng = self.sketchPrngs[sketchSize] if (sketchSize in self.sketchPrngs) else None
		# Given sequence length L and N mutated kmers, we consider the kmers
		# in A union B to be numbered from 0 to L+N-1, and we consider the
		# *un*mutated kmers to be the first L-N of these
		#    <---unmutated--> <---mutated, in A--> <---mutated, in B-->
		#   +----------------+--------------------+--------------------+
		#   | 0        L-1-N | L-N            L-1 | L            L+N-1 |
		#   +----------------+--------------------+--------------------+
		# The L-N *un*mutated kmers are A intersection B. The hash function
		# would effectively choose a random set of s of all L+N kmers as bottom
		# sketch BS(A union B), where s is the sketch size. So conceptually, we
		# have an urn with L+N balls, s of which are 'red'. We draw L-N balls
		# and want to know how many are red. This is the size of the
		# intersection of BS(A), BS(B), and BS(A union B).
		L = self.ntSequenceLength - (self.kmerSize-1)
		N = nMutated
		s = sketchSize
		if (N == L):    # hypergeom.rvs doesn't handle this case, a
			return 0    # .. case that seems perfectly legitimate
		nIntersection = hypergeom.rvs(L+N,s,L-N,random_state=prng)
		return nIntersection


# count_mutated_kmers_linear--
#	Given a sequence of errors, report the number of errors and mutated
#	kmers, treating the sequence as linear. The error sequence is a list of
#	1 (error) and 0 (non-error).
#
#	The algorithm maintains a sum of errors over a sliding window of length K.
#	wherever this sum is non-zero, the corresponding kmer is deemed to be
#	'mutated'.

def count_mutated_kmers_linear(kmerSize,errorSeq):
	nMutated = 0
	errorsInKmer = 0
	for pos in range(len(errorSeq)+1):
		# invariant: errorsInKmer is sum of errorSeq pos-kmerSize through pos-1
		if (pos >= kmerSize):
			if (errorsInKmer > 0): nMutated += 1   # pos-kmerSize is 'mutated'
			errorsInKmer -= errorSeq[pos-kmerSize]
		if (pos == len(errorSeq)): break
		errorsInKmer += errorSeq[pos]
	return nMutated


def count_mutated_kmers_linear_naive(kmerSize,errorSeq):
	seqLen = len(errorSeq) - (kmerSize-1)
	nMutated = 0
	for pos in range(seqLen):
		errorsInKmer = sum(errorSeq[pos:pos+kmerSize])
		if (errorsInKmer > 0): nMutated += 1       # pos is 'mutated'
	return nMutated


# r1_simulations--
#	Run several simulations and report how often the observed r1 is within the
#	specified interval.
#
# We return a dictionary that maps sketch size to the corresponding success
# rate of r1 derived from a simulated sketch. An additional key, "no sketch",
# maps the success rate of r1 derived from the number of mutated kmers.

def r1_simulations(numSimulations,L,k,r1,sketchSizes,r1Low,r1High,prngSeed=None,reportProgress=None):
	mutationModel = MutationModel(L,k,r1,sketchSizes=sketchSizes,prngSeed=prngSeed)

	counts = {"no sketch":0}
	if (sketchSizes != None):
		for sketchSize in sketchSizes:
			counts[sketchSize] = 0

	for trialNum in range(numSimulations):
		if (reportProgress != None):
			if (1+trialNum == 1) or ((1+trialNum) % reportProgress == 0):
				print("simulating trial %d" % (1+trialNum))

		mutationModel.generate()
		(_,nMutated) = mutationModel.count()
		qObs = nMutated / L
		r1Obs = q_to_r1(k,qObs)
		if (r1Low <= r1Obs <= r1High):
			counts["no sketch"] += 1

		if (sketchSizes != None):
			for sketchSize in sketchSizes:
				nIntersection = mutationModel.simulate_sketch(nMutated,sketchSize)
				jObs = nIntersection / sketchSize
				r1Obs = jaccard_to_r1(k,jObs)
				if (r1Low <= r1Obs <= r1High):
					counts[sketchSize] += 1

	successRate = {}
	for key in counts:
		successRate[key] = counts[key]/numSimulations
	return successRate


# nmut_simulations--
#	Run several simulations and report how often the observed nMut is within
#	the specified interval.
#
# We return a dictionary that maps sketch size to the corresponding success
# rate of nMut derived from a simulated sketch. An additional key, "no sketch",
# maps the success rate of nMut derived from the number of mutated kmers.

def nmut_simulations(numSimulations,L,k,nMut,sketchSizes,nMutLow,nMutHigh,prngSeed=None,reportProgress=None):
	q = nMut / L
	r1 = q_to_r1(k,q)
	mutationModel = MutationModel(L,k,r1,sketchSizes=sketchSizes,prngSeed=prngSeed)

	counts = {"no sketch":0}
	if (sketchSizes != None):
		for sketchSize in sketchSizes:
			counts[sketchSize] = 0

	for trialNum in range(numSimulations):
		if (reportProgress != None):
			if (1+trialNum == 1) or ((1+trialNum) % reportProgress == 0):
				print("simulating trial %d" % (1+trialNum))

		mutationModel.generate()
		(_,nMutated) = mutationModel.count()
		if (nMutLow <= nMutated <= nMutHigh):
			counts["no sketch"] += 1

		if (sketchSizes != None):
			for sketchSize in sketchSizes:
				nIntersection = mutationModel.simulate_sketch(nMutated,sketchSize)
				jObs = nIntersection / sketchSize
				nMutObs = L * jaccard_to_q(k,jObs)
				if (nMutLow <= nMutObs <= nMutHigh):
					counts[sketchSize] += 1

	successRate = {}
	for key in counts:
		successRate[key] = counts[key]/numSimulations
	return successRate


# jaccard_simulations--
#	Run several simulations and report how often the observed jaccard is within
#	the specified interval.
#
# We return a dictionary that maps sketch size to the corresponding success
# rate of jaccard derived from a simulated sketch. An additional key, "no
# sketch", maps the success rate of jaccard derived from the number of mutated
# kmers.

def jaccard_simulations(numSimulations,L,k,jaccard,sketchSizes,jLow,jHigh,prngSeed=None,reportProgress=None):
	r1 = jaccard_to_r1(k,jaccard)
	mutationModel = MutationModel(L,k,r1,sketchSizes=sketchSizes,prngSeed=prngSeed)

	counts = {"no sketch":0}
	if (sketchSizes != None):
		for sketchSize in sketchSizes:
			counts[sketchSize] = 0

	for trialNum in range(numSimulations):
		if (reportProgress != None):
			if (1+trialNum == 1) or ((1+trialNum) % reportProgress == 0):
				print("simulating trial %d" % (1+trialNum))

		mutationModel.generate()
		(_,nMutated) = mutationModel.count()
		qObs = nMutated / L
		jObs = q_to_jaccard(qObs)
		if (jLow <= jObs <= jHigh):
			counts["no sketch"] += 1

		if (sketchSizes != None):
			for sketchSize in sketchSizes:
				nIntersection = mutationModel.simulate_sketch(nMutated,sketchSize)
				jObs = nIntersection / sketchSize
				if (jLow <= jObs <= jHigh):
					counts[sketchSize] += 1

	successRate = {}
	for key in counts:
		successRate[key] = counts[key]/numSimulations
	return successRate

