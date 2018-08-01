bm.file.good = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters7.txt",
  header = TRUE,
  sep = ";"
)
bm.file.good$used.memory.centered = bm.file.good$used.memory - bm.file.good$used.memory[1]
bm.file.good$second = bm.file.good$second / 60

bm.file.bad = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters7.txt",
  header = TRUE,
  sep = ";"
)
bm.file.bad$used.memory.centered = bm.file.bad$used.memory - bm.file.bad$used.memory[1]
bm.file.bad$second = bm.file.bad$second / 60

bm.compboost = bm.file.good[8:489, ]
bm.mboost    = bm.file.good[490:nrow(bm.file.good), ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"
bm.compboost = bm.compboost[bm.compboost$used.memory.centered < 420 & bm.compboost$used.memory.centered > 0, ]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 788 & bm.mboost$used.memory.centered > 0, ]

bm.algos.good = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.bad[513:nrow(bm.file.bad), ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 200 & bm.mboost$used.memory.centered > 0, ]

bm.algos.bad = rbind(bm.compboost, bm.mboost)

plotMemResults(bm.algos.good, bm.algos.bad, "Memory Comparison using P-Spline Base-Learner with 5000 Iterations",
  "Elapsed Minutes", 0, 800)
