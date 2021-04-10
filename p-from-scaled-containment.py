#!/usr/bin/env python3

import sys
from math import ceil
import kmer_mutation_formulas_thm5 as thm5
import hypergeometric_slicer as hgslicer
from scipy.optimize import brentq, fsolve, newton
from scipy.stats import norm as scipy_norm
from numpy import sqrt

import argparse

try:
	from mpmath import mp as mpmath,mpf
	mpmath.dps = 50
except ModuleNotFoundError:
	mpf = lambda v:float(v)


def probit(p):
    return scipy_norm.ppf(p)

def compute_confidence_intervals(scaledContainmentsObserverved, L, k, alpha, s, debug=False):
    z_alpha = probit(1-alpha/2)
    f1 = lambda Nm: 1-1.0*Nm/L + z_alpha*sqrt( 1.0*Nm*(L-Nm)*(1-s)/(s * L**3) ) - Cks
    f1_mpf = lambda Nm: mpf(f1(Nm))
    f2 = lambda Nm: 1-1.0*Nm/L - z_alpha*sqrt( 1.0*Nm*(L-Nm)*(1-s)/(s * L**3) ) - Cks
    f2_mpf = lambda Nm: mpf(f2(Nm))

    all_results = []
    for (CksIx,Cks) in enumerate(scaledContainmentsObserverved):
        Nm_guess = L*(1-Cks)
        sol1_mpf = newton(f1_mpf, Nm_guess)
        sol2_mpf = newton(f2_mpf, Nm_guess)
        #if debug:
            #print( sol1, f1(sol1) )
             #print( float(sol1_new), float(f1_new(float(sol1_new))) )
             #print( float(sol1_new), float(f1(sol1_new)) )
             #print( sol1_mpf, f1_mpf(sol1_mpf) )
             #print( sol2, f2(sol2) )
             #print( sol2_mpf, f2_mpf(sol2_mpf) )

        sol1 = sol1_mpf
        sol2 = sol2_mpf

        Clow = 1-1.0*sol1/L
        Chigh = 1-1.0*sol2/L

        f3 = lambda pest: mpf((1-pest)**k + z_alpha*sqrt( thm5.var_n_mutated(L,k,pest) ) / L - Clow)
        f4 = lambda pest: mpf((1-pest)**k - z_alpha*sqrt( thm5.var_n_mutated(L,k,pest) ) / L - Chigh)

        phigh = newton(f3, Clow)
        plow = newton(f4, Chigh)

        #phigh = brentq(f3, 0.0001, 0.95)
        #plow = brentq(f4, 0.0001, 0.95)

        #print(phigh, f3(phigh))
        #print(plow, f4(plow))

        values = [L,k,confidence,Cks,Clow,Chigh,plow,phigh]
        all_results.append(values)
    return values

def main(args):
    global reportProgress,debug

    # parse the command line
    scaledContainmentsObserverved= []
    ntSequenceLength=None
    kmerSequenceLength=None
    scaledContainmentsObserverved += list(map(parse_probability, args.sccon))
    if args.length:
        ntSequenceLength = int_with_unit(args.length)
    if args.num_unique_kmers:
        kmerSequenceLength = int_with_unit(args.num_unique_kmers)
    kmerSize = args.ksize
    scaleFactor = parse_probability(args.scaled)
    confidence = parse_probability(args.confidence)
    prngSeed = args.seed # not used anywhere?
    debug = []
    if args.debug:
        debug = ["debug"]
    if args.debug_options:
       debug += args.debug_options

    # check for necessary info
    if (ntSequenceLength != None) and (kmerSequenceLength != None):
        if (kmerSequenceLength != ntSequenceLength + kmerSize-1):
            usage("nucleotide and kmer sequence lengths are inconsistent\nyou only need to specify one of them")
    elif (kmerSequenceLength != None):
        ntSequenceLength = kmerSequenceLength + (kmerSize-1)
    elif (ntSequenceLength == None):
        ntSequenceLength = 100000 + (kmerSize-1)

    # handle debugging options:
    if ("nocache" in debug):
        hgslicer.useCache = False
    if ("nojmonotonicity" in debug):
        hgslicer.doJMonotonicityCheck = False
    else:
        hgslicer.doJMonotonicityCheck = True
    if ("nsanity" in debug):
        hgslicer.doNLowSanityCheck  = True
        hgslicer.doNHighSanityCheck = True


    # compute the confidence interval(s)
    L = ntSequenceLength - (kmerSize-1)
    k = kmerSize
    alpha = 1 - confidence
    s = scaleFactor
    conf_intervals = compute_confidence_intervals(scaledContainmentsObserverved,L,k,alpha,s)

    #write results
    header = ["L","k","conf","Cks","CLow","CHigh","pLow","pHigh"]
    print("\t".join(header))
    for values in conf_intervals:
        print("\t".join(str(v)[:7] for v in values))


# parse_probability--
#	Parse a string as a probability

def parse_probability(s,strict=True):
    scale = 1.0
    if not isinstance(s, float):
        if (s.endswith("%")):
            scale = 0.01
            s = s[:-1]
        try:
            p = float(s)
        except:
            try:
                (numer,denom) = s.split("/",1)
                p = float(numer)/float(denom)
            except:
                raise ValueError
        p *= scale
    else:
        p=s
    if (strict) and (not 0.0 <= p <= 1.0):
        raise ValueError
    return p


# int_with_unit--
# Parse a string as an integer, allowing unit suffixes

def int_with_unit(s):
	if (s.endswith("K")):
		multiplier = 1000
		s = s[:-1]
	elif (s.endswith("M")):
		multiplier = 1000 * 1000
		s = s[:-1]
	elif (s.endswith("G")):
		multiplier = 1000 * 1000 * 1000
		s = s[:-1]
	else:
		multiplier = 1

	try:               return          int(s)   * multiplier
	except ValueError: return int(ceil(float(s) * multiplier))



def cmdline(sys_args):
    "Command line entry point w/argparse action."
    p = argparse.ArgumentParser(description="Compute confidence interval for the mutation rate p, given the observed number of mutated k-mers")
    p.add_argument("--sccon", nargs="+", help="observed number of mutated k-mers", required=True) # at least one observation is required

    # add # nucleotides and the # of unique k-mers as a mutually exclusive group, with one (and only one) required
    seqlen_info = p.add_mutually_exclusive_group(required=True) #, help="nucleotide and kmer sequence lengths are inconsistent\nyou only need to specify one of them")
    seqlen_info.add_argument("--length", type=int, help="number of nucleotides in the sequence") #default=1000000
    seqlen_info.add_argument("-L", "--num-unique-kmers", help="number of unique k-mers in the sequence")
    # optional arguments
    p.add_argument("--scaled", help="scaling factor of the sketch", default=0.1)
    p.add_argument("-k", "--ksize", type=int, default=21, help="kmer size")
    p.add_argument("-c", "--confidence", default=0.95, help="size of confidence interval, (value between 0 and 1)") # type=float would constrain to float
    p.add_argument("-s", "--seed", type=int, help="random seed for simulations")
    p.add_argument("--debug", action="store_true", help="debug")
    p.add_argument("--debug_options", nargs="*", help="additional options for debugging ('nocache', 'nojmonotonicity', 'nsanity')")
    args = p.parse_args()
    return main(args)

if __name__ == '__main__':
    returncode = cmdline(sys.argv[1:])
    sys.exit(returncode)
