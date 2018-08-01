bm.file.good = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters13.txt",
  header = TRUE,
  sep = ";"
)
bm.file.good$used.memory.centered = bm.file.good$used.memory - bm.file.good$used.memory[1]
bm.file.good$second = bm.file.good$second / 60

bm.file.bad = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters14.txt",
  header = TRUE,
  sep = ";"
)
bm.file.bad$used.memory.centered = bm.file.bad$used.memory - bm.file.bad$used.memory[1]
bm.file.bad$second = bm.file.bad$second / 60

bm.compboost = bm.file.good[6:411, ]
bm.mboost    = bm.file.good[424:2134, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (disabling sparse matrices)"

bm.algos.linear = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.bad[421:2131, ]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (enabling sparse matrices)"

bm.algos.linear = rbind(bm.algos.linear, bm.mboost)


bm.file.good = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters15.txt",
  header = TRUE,
  sep = ";"
)
bm.file.good$used.memory.centered = bm.file.good$used.memory - bm.file.good$used.memory[1]
bm.file.good$second = bm.file.good$second / 60

bm.file.bad = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters16.txt",
  header = TRUE,
  sep = ";"
)
bm.file.bad$used.memory.centered = bm.file.bad$used.memory - bm.file.bad$used.memory[1]
bm.file.bad$second = bm.file.bad$second / 60

bm.compboost = bm.file.good[10:1742, ]
bm.mboost    = bm.file.good[1744:nrow(bm.file.good), ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (disabling sparse matrices)"

bm.algos.spline = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.bad[6:7060, ]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (enabling sparse matrices)"

bm.algos.spline = rbind(bm.algos.spline, bm.mboost)



plotMemResults(bm.algos.linear, bm.algos.spline, "Memory Comparison using 50000 Observations",
  "Elapsed Minutes", 0, 20000)
