cat("Simulate data\n")

set.seed(31415)
nsim = 5 * 1e5
ncat = 20L
x = sample(LETTERS[seq_len(ncat)], nsim, TRUE)
y = rnorm(nsim)

mydata = data.frame(y = y, x = x)
mstop = 1000L

