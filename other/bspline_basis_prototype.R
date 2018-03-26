# ============================================================================ #
#                                                                              #
#                   Prototype for creating B(asic)-Spline Basis                #
#                        Code is based on the Nurbs Book                       #
#                                                                              #
# ============================================================================ #

# For an application and comparison with splines::splineDesign see 
# `compare_bspline_prototype_mboost_bbs.R`

## Binary search (Site 68):

# function to find first position where min {i : u <= U[i]}
findSpan = function (u, U)
{
  m = length(U)
  
  # Special cases:
  if (u < U[2]) { return (1) }
  
  low  = 1
  high = m
  mid  = round((low + high) / 2)
  
  while (u < U[mid] || u >= U[mid + 1])
  {
    if (u < U[mid]) { 
      high = mid  
    } else {
      low = mid    
    }
    mid = round((low + high) / 2)
  }
  # smallest possible number in R is 1, in C++ we have to switch everything
  # by -1!
  return (mid)
}

# # Example:
# p = 2
# U = c(0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5)
# u = 5/2
# 
# findSpan(p, u, U)

## Create Base (Site 70):

# Important:
#   - Check a priori if u is within the range of the originally used data
#   - One preprocessing step is used here (expanding the knots), see 
#     `compare_bspline_prototype_mboost_bbs.R`

basisFuns = function (i, u, p, U)
{
  # full base is length of knots minus the number of coefficients
  full.base = rep(0, length(U) - (p + 1))
  if (i > length(full.base)) { i = length(full.base) }
  # if (i <= p) {
  #   i = p + 1
  # }
  
  # if (u <= U[1]) { 
  #   full.base[1] = 1
  #   return (full.base) 
  # }
  # if (u >= U[length(U)]) { 
  #   full.base[length(full.base)] = 1
  #   return (full.base) 
  # }
  
  # Output for basis functions:
  N = numeric(length = p + 1)
  right = left = numeric(length = p)
  
  # In C++ initialization with N[0]
  N[1] = 1.0
  
  for (j in 1:p) {
    
    left[j]  = u - U[i + 1 - j]
    right[j] = U[i + j] - u
    
    saved = 0
    
    for (r in 0:(j - 1)) {
      temp = N[r + (1)] / (right[r + 1] + left[j - r])
      N[r + (1)] = saved + right[r + 1] * temp
      saved = left[j - r] * temp
    }
    N[j + (1)] = saved
  }
  full.base[((i - p):i) ] = N
  
  return (full.base)
}

# - Efficient 
# - Guarantees no division by 0

# # Example:
# i = findSpan(p, u, U)
# basisFuns(i, u, p, U)
# 
# 
# 
# p = 3
# U = seq(0, 10, length.out = 11)
# u = 4.2
# 
# basisFuns(findSpan(p, u, U), u, p, U)
# 
# 
# x = runif(100, -5, 7)
# y = 2 * x + 1/5 * x^2 - 1/10 * x^3 + rnorm(100, 0, 2)
# 
# plot(x, y)
# 
# U = seq(min(x), max(x), length.out = 10)
# p = 2
# 
# abline(v = U)
# 
# u = x[10]
# basisFuns(findSpan(p, u, U), u, p, U)
# 
# spline.base = as.matrix(lapply(x, function (u) {
#   basisFuns(findSpan(p, u, U), u, p, U)
# }))
