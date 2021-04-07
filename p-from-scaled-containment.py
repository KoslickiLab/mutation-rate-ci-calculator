#!/usr/bin/env python3

from sys  import argv,exit
from math import ceil
import kmer_mutation_formulas_thm5 as thm5
import hypergeometric_slicer as hgslicer
from scipy.optimize import brentq, fsolve, newton
from scipy.stats import norm as scipy_norm
from numpy import sqrt

try:
	from mpmath import mp as mpmath,mpf
	mpmath.dps = 50
except ModuleNotFoundError:
	mpf = lambda v:float(v)

def usage(s=None):
    message = """
Compute confidence interval for the mutation rate p, given the observed number
of mutated k-mers.

usage: p-from-scaled-containment.py [options]
  --sccon=<list>              (Cks=) (cumulative) observed number of mutated
                              k-mers; <list> is a comma-separated list of
                              numbers
  --scale=<probability>       (s=) scaling factor of the hash
  
  --length=<N>                (l=) sequence length (number of NUCLEOTIDES in
                              the sequence)
                              (default is 1000 plus kmer size minus 1)
  L=<N>                       (L=) sequence length (number of KMERS in
                              the sequence)
                              (default is 1000000)
  --k=<N>                     (K=) kmer size
                              (default is 21)
  --confidence=<probability>  (C=) size of confidence interval
                              (default is 95%)
  --seed=<string>             random seed for simulations"""
  
    if (s == None): exit (message)
    else:           exit ("%s\n%s" % (s,message))


def probit(p):
    return scipy_norm.ppf(p)

def main():
    global reportProgress,debug
    # parse the command line
    scaledContainmentsObserverved  = []
    scaleFactor        = 0.1 #default 
    ntSequenceLength   = None
    kmerSequenceLength = None
    kmerSize           = 21
    confidence         = 0.95
    prngSeed           = None
    debug              = []
    for arg in argv[1:]:
        if ("=" in arg):
            argVal = arg.split("=",1)[1]
        if (arg in ["--help","-help","--h","-h"]):
            usage()
        elif (arg.lower().startswith("--sccon=")) or (arg.upper().startswith("Cs=")):
            scaledContainmentsObserverved += list(map(parse_probability,argVal.split(",")))
        elif (arg.startswith("--length=")) or (arg.startswith("l=")):
            ntSequenceLength = int_with_unit(argVal)
        elif (arg.startswith("--scale=")) or (arg.startswith("s=")):
            scaleFactor = parse_probability(argVal)
        elif(arg.startswith("L=")):
            kmerSequenceLength = int_with_unit(argVal)
        elif (arg.startswith("--kmer=")) or (arg.upper().startswith("K=")):
            kmerSize = int(argVal)
        elif (arg.startswith("--confidence=")) or (arg.startswith("C=")):
            confidence = parse_probability(argVal)
        elif (arg.startswith("--seed=")):
            prngSeed = argVal
        elif (arg == "--debug"):
            debug += ["debug"]
        elif (arg.startswith("--debug=")):
            debug += argVal.split(",")
        elif (arg.startswith("--")):
            usage("unrecognized option: %s" % arg)
        else:
            usage("unrecognized option: %s" % arg)
    if (scaledContainmentsObserverved == []):
        usage("you have to give me at least one scaled containment observation")
    if (ntSequenceLength != None) and (kmerSequenceLength != None):
        if (kmerSequenceLength != ntSequenceLength + kmerSize-1):
            usage("nucleotide and kmer sequence lengths are inconsistent\nyou only need to specify one of them")
    elif (kmerSequenceLength != None):
        ntSequenceLength = kmerSequenceLength + (kmerSize-1)
    elif (ntSequenceLength == None):
        ntSequenceLength = 100000 + (kmerSize-1)
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
    
    header = ["L","k","conf","Cks","CLow","CHigh","pLow","pHigh"]
    print("\t".join(header))
    z_alpha = probit(1-alpha/2)
    f1 = lambda Nm: 1-1.0*Nm/L + z_alpha*sqrt( 1.0*Nm*(L-Nm)*(1-s)/(s * L**3) ) - Cks
    f1_mpf = lambda Nm: mpf(f1(Nm))
    f2 = lambda Nm: 1-1.0*Nm/L - z_alpha*sqrt( 1.0*Nm*(L-Nm)*(1-s)/(s * L**3) ) - Cks
    f2_mpf = lambda Nm: mpf(f2(Nm))
    
    for (CksIx,Cks) in enumerate(scaledContainmentsObserverved):
        Nm_guess = L*(1-Cks)
        sol1_mpf = newton(f1_mpf, Nm_guess)
        sol2_mpf = newton(f2_mpf, Nm_guess)
        
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
        print("\t".join(str(v)[:7] for v in values))
        
            
# parse_probability--
#	Parse a string as a probability

def parse_probability(s,strict=True):
    scale = 1.0
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

if __name__ == "__main__": main()
