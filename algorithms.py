
# Similarity Algorithms

from math import sqrt

def correlation(size, dotproduct, ratingsum, rating2sum, ratingnormsqrd, rating2normsqrd):
	
	# correlation between two vectors A, B is 
	# 	[n * dotProduct(A, B) - sum(A) * sum(B)] / {sqrt(n * norm(A)^2 - sum(A)^2) * sqrt(n * norm(B)^2 - sum(B)^2)}

	numerator = size * dotproduct - ratingsum * rating2sum
	denominator = sqrt(size * ratingnormsqrd - ratingsum * ratingsum) * sqrt(size * rating2normsqrd - rating2sum * rating2sum)

	return (numerator / float(denominator)) if denominator else 0.0

def jaccard(usersincommon, totalusers1, totalusers2):
	
	# jaccard similarity between vecotr A, B is
	#	intersection(A, B)/union(A, B)
	
	union = totalusers1 + totalusers2 - usersincommon

	return (usersincommon / (float(union))) if union else 0.0

def norm_correlation(size, dotproduct, ratingsum, rating2sum, ratingnormsqrd, rating2normsqrd):
	
	# The correlation between two vectors A, B is
     	#	 cov(A, B) / (stdDev(A) * stdDev(B))
     	#	 The normalization is to give the scale between [0,1].

	similarity = correlation(size, dotproduct, ratingsum, rating2sum, ratingnormsqrd, rating2normsqrd)
	
	return (similarity + 1.0) / 2.0

def cosine(dotproduct, ratingnormsqrd, rating2normsqrd):
	
	# cosine between two vectors A, B
	#	dotproduct(A, B) / (norm(A) * norm(B))

	numerator = dotproduct
	denominator = ratingnormsqrd * rating2normsqrd
	
	return (numerator / (float(denominator))) if denominator else 0.0

def reg_correlation(size, dotproduct, ratingsum, rating2sum, ratingnormsqrd, rating2normsqrd, virtcount, priorcorrelation):
	
	# regularized correlation between vectors A, B
	#	 RegularizedCorrelation = w * ActualCorrelation + (1 - w) * PriorCorrelation
        #  		where w = # actualPairs / (# actualPairs + # virtualPairs).

	unregularized = correlation(size, dotproduct, ratingsum, rating2sum, ratingnormsqrd, rating2normsqrd)
	
	w = size / float(size + virtcount)

	return w * unregularized + (1.0 - w) * priorcorrelation

def combinations(iterable, r):
	
    """
    Implementation of itertools combinations method. Re-implemented here because
    of import issues in Amazon Elastic MapReduce. Was just easier to do this than
    bootstrap.
    More info here: http://docs.python.org/library/itertools.html#itertools.combinations
    Input/Output:
    combinations('ABCD', 2) --> AB AC AD BC BD CD
    combinations(range(4), 3) --> 012 013 023 123

    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)
