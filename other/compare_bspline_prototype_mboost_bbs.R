# ============================================================================ #
#                                                                              #
#                   Prototype for creating B(asic)-Spline Basis                #
#                   Functions are from `bspline_basis_prototype`               #
#                                                                              #
# ============================================================================ #


x <- c(1, seq(0, 11, length.out = 5), 10.9)

n = 100
x1 = x

n.knots = 30
degree = 3 # mboost default

spline1 = mboost::bbs(x1, knots = n.knots, df = 4, boundary.knots = range(x), 
  degree = degree)
bb.mboost = mboost::extract(spline1, "design")

knots = attr(bb.mboost, "knots")

# This is what mboost does to get every 
myknots = seq(min(x), max(x), length.out = n.knots + 2)

knot.range = diff(myknots)[1]
myknots = c(min(x) - 3:1 * knot.range, myknots, max(x) + 1:3 * knot.range)

# Knot the same as mboost (degree has to be degree + 1 since splineDesign uses
# the number of coefficent as ord):
bb = splines::splineDesign(myknots, x = x1, outer.ok = TRUE, ord = degree + 1)
bb.m = mboost:::bsplines(x1, knots, boundary.knots = range(x1), degree = degree)

bb
bb.m
bb.mboost

attributes(bb.m) = NULL
attr(bb.m, "dim") = dim(bb)

all.equal(bb, bb.m)




# My stupid but easy algorithm (prototype):

idx.test = 6

u = x[idx.test]

idx = findSpan(u = u, U = myknots)
basisFuns(idx, u = u, p = degree, U = myknots)

bb[idx.test, ]


mybb = matrix(0, nrow = length(x), ncol = length(myknots) - (degree + 1))
  
for (i in seq_len(nrow(mybb))) {
  idx.mybb = findSpan(u = x[i], U = myknots)
  mybb[i, ] = basisFuns(idx.mybb, u = x[i], p = degree, U = myknots)
}
  
all.equal(mybb, bb)
