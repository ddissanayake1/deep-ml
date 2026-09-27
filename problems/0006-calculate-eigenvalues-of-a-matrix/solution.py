import numpy
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

	# det (a-l, b)
	#	  (c, d-l)
	# (a-l)*(d-l) - b*c
	# a*d - l*(a+d) + l**2 - b*c
	# l**2 - l*(a+d) + (a*d - b*c)
	# coeffs = 1, (a+d), (a*d - b*c)

	a,b = matrix[0]
	c,d = matrix[1]

	eigenvalues = numpy.roots([1, -(a+d), (a*d-b*c)])
	return eigenvalues