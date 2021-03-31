#!/usr/bin/env python3
"""
Formulas relating to kmers in strings subjected to independent/uniform
nucleotide substitutions, and the "Nmut hypothesis test theorem"/"Nmut
confidence interval theorem", as described in "The statistics of kmers from a
sequence undergoing a simple mutation process without spurious matches,"
Blanca, Harris, Koslicki and Medvedev.

The underlying theory is described in theorem 5 in the bioRxiv version of the
manuscript at https://www.biorxiv.org/content/10.1101/2021.01.15.426881v1

k:  Kmer length.
L:  Sequence length; specifically, the number of complete KMERS in the sequence.
    The corresponding nucleotide sequence length would be L+k-1.
r1: Nucleotide substitution rate.
q:  1-(1-r1)^k, the probability that a kmer is mutated, i.e. that a kmer
    contains a least one substitution.

For info on the brentq solver, which is used herein, see
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html"""

from sys            import stderr,exit
from math           import sqrt
from scipy.optimize import brentq
from scipy.stats    import norm as scipy_norm

try:
	from mpmath import mp as mpmath,mpf
	mpmath.dps = 50
except ModuleNotFoundError:
	mpf = lambda v:float(v)

#==========
# formulas for Nmutated
#==========

def p_mutated(k,r1):
	return r1_to_q(k,r1)

def r1_to_q(k,r1):
	#return 1-(1-r1)**k
	r1 = mpf(r1)
	q = 1-(1-r1)**k
	return float(q)


def p_mutated_inverse(k,q):
	return q_to_r1(k,r1)

def q_to_r1(k,q):
	if not (0 <= q <= 1): return float("nan")
	#return 1-(1-q)**(1.0/k)
	q = mpf(q)
	r1 = 1-(1-q)**(1.0/k)
	return float(r1)


def exp_n_mutated(L,k,r1):
	q = r1_to_q(k,r1)
	return L*q


def var_n_mutated(L,k,r1,q=None):
	# there are computational issues in the variance formula that we solve here
	# by the use of higher-precision arithmetic; the problem occurs when r is
	# very small; for example, with L=10,k=2,r1=1e-6 standard precision
	# gives varN<0 which is nonsense; by using the mpf type, we get the correct
	# answer which is about 0.000038
	if (r1 == 0): return 0.0
	r1 = float(r1)
	if (q == None): # we assume that if q is provided, it is correct for r1
		q = r1_to_q(k,r1)
	varN = L*(1-q)*(q*(2*k+(2/r1)-1)-2*k) \
	     + k*(k-1)*(1-q)**2 \
	     + (2*(1-q)/(r1**2))*((1+(k-1)*(1-q))*r1-q)
	assert (varN>=0.0)
	return float(varN)


# estimate_r1_from_n_mutated:
#   q = 1-(1-r1)^k  and e[nMutated] = qL
# so
#   qHat = nMutated/L
#   r1Est = 1-kth_root(1-qHat) = 1-kth_root(1-nMutated/L) 

def estimate_q_from_n_mutated(L,nMutated):
	return float(nMutated)/L


def estimate_r1_from_n_mutated(L,k,nMutated):
	return 1 - (1-float(nMutated)/L)**(1.0/k)


def confidence_interval_r1_from_n_mutated(L,k,r1,alpha):
	z = probit(1-alpha/2)
	q = r1_to_q(k,r1)
	varN = var_n_mutated(L,k,r1,q=q)
	(nLow,nHigh) = confidence_interval(L,q,varN,z)
	r1Low  = q_to_r1(k,nLow/L)
	r1High = q_to_r1(k,nHigh/L)
	return (r1Low,r1High)


def in_confidence_interval_q_from_n_mutated(L,k,r1,alpha,nMutatedObserved,useInverse=True):
	# nMutatedObserved argument can be a single value or a list
	if (not isinstance(nMutatedObserved,list)):
		nMutatedObserved = [nMutatedObserved]
	z = probit(1-alpha/2)
	q = r1_to_q(k,r1)

	if (useInverse):
		numInCI = 0
		for nMutated in nMutatedObserved:
			q1 = q_for_n_mutated_high(L,k,nMutated,z)   # nHigh(q1) == nMut
			q2 = q_for_n_mutated_low (L,k,nMutated,z)   # nLow (q2) == nMut
			if (q1 < q < q2):
				numInCI += 1
		return numInCI
	else:
		# we expect this to give exactly the same results as the useInverse case
		qLow  = n_low (L,k,q,z) / L
		qHigh = n_high(L,k,q,z) / L
		numInCI = 0
		for nMutated in nMutatedObserved:
			qHat = float(nMutated) / L
			if (qLow < qHat < qHigh):
				numInCI += 1
		return numInCI


# q_for_n_mutated_high--
#	find q s.t. nHigh(q) == nMut
#
# Note: nMut==0 is a special case. When q=0 the formula for variance has a zero
# in the denominator and thus fails to compute. However, the limit of that
# formula as q goes to zero is zero (and in fact, it is easy to see that the
# variance is truly zero when q=0). This means the formulas for nLow and nHigh,
# e.g. L*q-z*sqrt(varN), are zero when q=0. Thus if nMut=0, 0 is the q for
# which nHigh(q) == nMut.
#
# nMut==L is another special case. There are two solutions in this case, one
# of which is q=1. We are interested in the other solutions

def q_for_n_mutated_high(L,k,nMut,z,checkDerivative=True):
	if (nMut == 0): return 0.0   # special case, see note above
	qRight = 1 if (nMut<L) else 1-1e-5
	qLeft = 1e-5
	attemptsLeft = 10
	while (n_high(L,k,qLeft,z) >= nMut):
		qLeft /= 2
		attemptsLeft -= 1
		if (attemptsLeft < 0): break
	if (n_high(L,k,qLeft,z) >= nMut):
		# this is just laziness, it really means our assumptions about the
		# solution space are wrong
		print("q_for_n_mutated_high(L=%s,k=%d,nMut=%s)" % (L,k,nMut))
		print("n_high(...,qLeft=%s)=%s" % (qLeft,n_low(L,k,qLeft,z)))
		raise ValueError

	# at this point,
	#	n_high(L,k,qLeft,z)  - nMut < 0
	#	n_high(L,k,qRight,z) - nMut > 0
	# so we can use the Brent's method to find the solution in the bracketed
	# interval
	func = lambda q: n_high(L,k,q,z)-nMut
	qSoln = brentq(func,qLeft,qRight)

	if (checkDerivative) and (qSoln != 1):
		# limit(dNHigh) as q->1 appears to be non-negative
		alpha = 2*(1-inverse_probit(z))    # because z = probit(1-alpha/2)
		dNHigh = n_high_derivative(L,k,qSoln,alpha)
		#print (("for nHigh(q)=%s (for L=%d k=%d) d(nHigh)/dr at q=%.9f is %9f") \
		#     % (nMut,L,k,qSoln,dNHigh),file=stderr)
		assert (dNHigh > 0.0), \
		       ("solution of nHigh(q)=%s (for L=%d k=%d) fails derivative test" 
		      + "\nd(nHigh)/dr at q=%.9f is %9f") \
		     % (nMut,L,k,qSoln,dNHigh)

	return qSoln


# q_for_n_mutated_low--
#	find q s.t. nLow(q) == nMut
#
# Note: nMut==L is a special case. When q=1 all kmers are mutated and thus
# var(nMut)=0. The formula for nLow then simplifies as
#	nLow = Lq - z*sqrt(varN) = L*q = L
# Moreover, if q<1 then varN>0 and nLow < Lq < L.  Thus if nMut=L, 1 is the q
# for which nLow(q) == nMut.

def q_for_n_mutated_low(L,k,nMut,z,checkDerivative=True):
	if (nMut == L): return 1.0   # special case, see note above
	qRight = 1
	qLeft = 1e-5
	attemptsLeft = 10
	while (n_low(L,k,qLeft,z) >= nMut):
		qLeft /= 2
		attemptsLeft -= 1
		if (attemptsLeft < 0): break
	if (n_low(L,k,qLeft,z) >= nMut):
		# this is just laziness, it really means our assumptions about the
		# solution space are wrong
		print("q_for_n_mutated_low(L=%s,k=%d,nMut=%s)" % (L,k,nMut))
		print("n_low(...,qLeft=%s)=%s" % (qLeft,n_low(L,k,qLeft,z)))
		raise ValueError

	# at this point,
	#	n_low(L,k,qLeft,z)  - nMut < 0
	#	n_low(L,k,qRight,z) - nMut > 0
	# so we can use the Brent's method to find the solution in the bracketed
	# interval
	func = lambda q: n_low(L,k,q,z)-nMut
	qSoln = brentq(func,qLeft,qRight)

	if (checkDerivative):
		# limit(dNLow) as q->1 appears to be negative; note that this assert
		# should *never* trigger, since we handled nMut==L as a special case
		# and thus the solver should never produce qSoln==1
		assert (qSoln != 1), \
		       ("solution of nLow(q)=%s (for L=%d k=%d) fails derivative test" 
		      + "\nd(nLow)/dr at q=%.9f is negative") \
		     % (nMut,L,k,qSoln)

		alpha = 2*(1-inverse_probit(z))    # because z = probit(1-alpha/2)
		dNLow = n_low_derivative(L,k,qSoln,alpha)
		#print (("for nLow(q)=%s (for L=%d k=%d) d(nLow)/dr at q=%.9f is %9f") \
		#     % (nMut,L,k,qSoln,dNLow),file=stderr)
		assert (dNLow > 0.0), \
		       ("solution of nLow(q)=%s (for L=%d k=%d) fails derivative test" 
		      + "\nd(nLow)/dr at q=%.9f is %9f") \
		     % (nMut,L,k,qSoln,dNLow)

	return qSoln


# confidence interval for Nmutated

def n_low(L,k,q,z):
	r1 = q_to_r1(k,q)
	varN = var_n_mutated(L,k,r1)
	return L*q - z*sqrt(varN)

def n_high(L,k,q,z):
	r1 = q_to_r1(k,q)
	varN = var_n_mutated(L,k,r1)
	return L*q + z*sqrt(varN)

# derivative of n_low and n_high w.r.t. r1

def n_low_derivative(L,k,q,alpha):
	r1 = q_to_r1(k,q)
	z = probit(1-alpha/2)
	derivativeMiddle = (k*L) if (r1==1) else ((k*L*(1-q))/(1-r1))
	derivativeOffset = \
	    ((1-q)*(4*q+r1*(-6+2*k+2*r1-2*k*r1+L*(-2-2*(-1+k)*r1+k*r1**2)+2*(1-q)*(3-3*k*(1-r1)-r1-k**3*r1**2+k**2*(-2*r1+r1**2)+L*(1-r1+k*(2-r1)*r1+2*k**2*r1**2))))) \
	     / (2*(1-r1)*r1**2*sqrt((1-q)*(-2*(1-r1)+L*(2-r1)*r1)+(1-q)**2*(2*(1-r1)+k*(2-r1)*r1+k**2*r1**2+L*(-2*r1+r1**2-2*k*r1**2))))
	return derivativeMiddle - z*derivativeOffset


def n_high_derivative(L,k,q,alpha):
	r1 = q_to_r1(k,q)
	z = probit(1-alpha/2)
	derivativeMiddle = (k*L) if (r1==1) else ((k*L*(1-q))/(1-r1))
	derivativeOffset = \
	    ((1-q)*(4*q+r1*(-6+2*k+2*r1-2*k*r1+L*(-2-2*(-1+k)*r1+k*r1**2)+2*(1-q)*(3-3*k*(1-r1)-r1-k**3*r1**2+k**2*(-2*r1+r1**2)+L*(1-r1+k*(2-r1)*r1+2*k**2*r1**2))))) \
	     / (2*(1-r1)*r1**2*sqrt((1-q)*(-2*(1-r1)+L*(2-r1)*r1)+(1-q)**2*(2*(1-r1)+k*(2-r1)*r1+k**2*r1**2+L*(-2*r1+r1**2-2*k*r1**2))))
	return derivativeMiddle + z*derivativeOffset


# same derivative, just a different formula
#def n_low_derivative_old(L,k,q,alpha):
#	r1 = q_to_r1(k,q)
#	derivativeMiddle = (k*L*(1-q))/(1-r1)
#	derivativeOffset = \
#	    ((1-q)*(4*q-r1*(6-2*k-2*r1+2*k*r1+L*(2-2*r1+2*k*r1-k*r1**2)+2*(1-q)*(-3+3*k-L+(-1+2*k)*(-1+k-L)*r1+k*(k*(-1+k-2*L)+L)*r1**2)))) \
#	     / (sqrt(2)*(1-r1)*r1**2*sqrt((1-q)*(-2+2*r1+2*L*r1-L*r1**2+(1-q)*(2+r1*(2*k-2*(1+L)+(k*(-1+k-2*L)+L)*r1)))))
#	return derivativeMiddle - inverse_erf(1-alpha)*derivativeOffset
#
#
#def n_high_derivative_old(L,k,q,alpha):
#	r1 = q_to_r1(k,q)
#	derivativeMiddle = (k*L*(1-q))/(1-r1)
#	derivativeOffset = \
#	    ((1-q)*(4*q-r1*(6-2*k-2*r1+2*k*r1+L*(2-2*r1+2*k*r1-k*r1**2)+2*(1-q)*(-3+3*k-L+(-1+2*k)*(-1+k-L)*r1+k*(k*(-1+k-2*L)+L)*r1**2)))) \
#	     / (sqrt(2)*(1-r1)*r1**2*sqrt((1-q)*(-2+2*r1+2*L*r1-L*r1**2+(1-q)*(2+r1*(2*k-2*(1+L)+(k*(-1+k-2*L)+L)*r1)))))
#	return derivativeMiddle + inverse_erf(1-alpha)*derivativeOffset

#==========
# formulas for Nisland
#==========

def exp_n_island(L,k,r1):
	q = r1_to_q(k,r1)
	return L*r1*(1-q) + q - r1*(1-q)


def exp_n_island_max(L,k):
	# maximum value of E[Nisland]
	return 1 + float(L-2)/(k+1) * ((float(L-2)*k)/((L-1)*(k+1)))**k


def exp_n_island_argmax_r1(L,k):
	# value of r1 which maximizes E[Nisland]
	return float(L+k-1)/((L-1)*(k+1))


def var_n_island(L,k,r1):
	q = r1_to_q(k,r1)
	return L*r1*(1-q)*(1-r1*(1-q)*(2*k+1)) \
	     + (k**2)*(r1**2)*(1-q)**2 \
	     + k*r1*(3*r1+2)*(1-q)**2 \
	     + (1-q)*((1-q)*(r1**2)-q-r1)


# estimate_r1_from_n_island:
#	We want to find r1 s.t. exp_n_island(r1) == nIsland. Note that there are
#	usually two solutions.
#
# We look for solutions to f(r) = E[Nisland(L,k,r)] - nIsland == 0. Note that
# E[Nisland(L,k,0)] = 0 and E[Nisland(L,k,1)] = 1, and that the derivative of
# E[Nisland(L,k,r)] wrt r is positive at r=0, crosses zero at some 0<r'<1, and
# returns to zero at r=1. Thus E[Nisland] peaks at r' between 0 and 1. So long
# as the observed nIsland is more than 1 and less than this peak, we'll have
# one solution between 0 and r' and another between r' and 1.

def estimate_r1_from_n_island(L,k,nIsland):
	if (nIsland < 0): return ()
	if (nIsland == 0): return (0.0,)
	assert (nIsland >= 1)
	rPeak = exp_n_island_argmax_r1(L,k)
	if (nIsland >= exp_n_island_max(L,k)): return (rPeak,)
	# at this point,
	#	E[Nisland(L,k,0)     - nIsland < 0
	#	E[Nisland(L,k,rPeak) - nIsland > 0
	#	E[Nisland(L,k,1)     - nIsland < 0
	# so we can use the Brent's method to find solutions in those two bracketed
	# intervals
	func = lambda r1: exp_n_island(L,k,r1)-nIsland
	soln1 = brentq(func,0.0,rPeak)
	soln2 = brentq(func,rPeak,1.0)
	return (soln1,soln2)


def impossible_n_island(L,k,nIsland):
	if (nIsland < 0): return True
	if (nIsland == 0): return False
	return (nIsland >= exp_n_island_max(L,k))


def estimate_q_from_n_island(L,k,nIsland):
	r1Solutions = estimate_r1_from_n_island(L,k,nIsland)
	return map(lambda r1:r1_to_q(k,r1),r1Solutions)


def confidence_interval_r1_from_n_island(L,k,r1,alpha):
	z = probit(1-alpha/2)
	q = r1_to_q(k,r1)
	varN = var_n_island(L,k,r1)
	(nLow,nHigh) = confidence_interval(L,q,varN,z)
	r1Low  = q_to_r1(k,nLow/L)
	r1High = q_to_r1(k,nHigh/L)
	return (r1Low,r1High)


def in_confidence_interval_q_from_n_island(L,k,r1,alpha,nIslandObserved,nMutatedObserved,useInverse=True):
	return float("nan") # not implemented

#==========
# formulas relating to confidence intervals
#==========

# confidence_interval--

def confidence_interval(L,q,varN,z):
	ciMiddle = L*q
	ciHalfWidth  = z*sqrt(varN)
	return (ciMiddle-ciHalfWidth,ciMiddle+ciHalfWidth)


# probit--
#
# see https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p

def probit(p):
	return scipy_norm.ppf(p)

def inverse_probit(z):
	return scipy_norm.cdf(z)

def inverse_erf(alpha):
	return scipy_norm.ppf((1+alpha)/2.0) / sqrt(2)

