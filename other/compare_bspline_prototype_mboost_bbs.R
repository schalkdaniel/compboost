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
myknots = c(min(x) - degree:1 * knot.range, myknots, max(x) + 1:degree * knot.range)

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

source ("C:/Users/schal/OneDrive/github_repos/compboost/other/bspline_basis_prototype.R")

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

# C++ algorithm:
splines.cpp = "C:/Users/schal/OneDrive/github_repos/compboost/other/splines.cpp"

if (file.exists(splines.cpp)) {
  Rcpp::sourceCpp(file = splines.cpp, rebuild = TRUE)
}

knots = createKnots(values = x, n_knots = n.knots, degree = degree)
mybb.cpp = as.matrix(createBasis(values = x, degree = degree, knots = knots))

mydim = dim(mybb.cpp)
attributes(mybb.cpp) = NULL
dim(mybb.cpp) = mydim

all.equal(mybb.cpp, bb)


# small benchmark with r's splines library:
microbenchmark::microbenchmark(
  "My C++" = createBasis(values = x, degree = degree, knots = knots),
  "R splines" = splines::splineDesign(myknots, x = x, outer.ok = TRUE, ord = degree + 1), 
  times = 10L
)


