bm.file.good = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters9.txt",
  header = TRUE,
  sep = ";"
)
bm.file.good$used.memory.centered = bm.file.good$used.memory - bm.file.good$used.memory[1]
bm.file.good$second = bm.file.good$second / 60

bm.compboost = bm.file.good[7:34, ]
bm.mboost    = bm.file.good[198, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"

bm.mboost$algo = "mboost (disabling sparse matrices)"



bm.algos.linear = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.good[198, ]

bm.mboost$algo = "mboost (enabling sparse matrices)"

bm.algos.linear = rbind(bm.algos.linear, bm.mboost)




bm.compboost = bm.file.good[34:197, ]
bm.mboost    = bm.file.good[198, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"

bm.mboost$algo = "mboost (disabling sparse matrices)"

bm.algos.spline = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.good[198, ]

bm.mboost$algo = "mboost (enabling sparse matrices)"

bm.algos.spline = rbind(bm.algos.spline, bm.mboost)



plotMemResults(bm.algos.linear, bm.algos.spline, "Memory Comparison using 2000 Base-Learner",
  "Elapsed Minutes", 0, 700)
