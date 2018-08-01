bm.file.good = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters5.txt",
  header = TRUE,
  sep = ";"
)
bm.file.good$used.memory.centered = bm.file.good$used.memory - bm.file.good$used.memory[1]
bm.file.good$second = bm.file.good$second / 60

bm.file.bad = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters6.txt",
  header = TRUE,
  sep = ";"
)
bm.file.bad$used.memory.centered = bm.file.bad$used.memory - bm.file.bad$used.memory[1]
bm.file.bad$second = bm.file.bad$second / 60

bm.compboost = bm.file.good[3:79, ]
bm.mboost    = bm.file.good[79:792, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"
bm.compboost = bm.compboost[bm.compboost$used.memory.centered < 130 & bm.compboost$used.memory.centered >= 0, ]
bm.compboost = bm.compboost[! ((seq_along(bm.compboost$second) > 10) & (bm.compboost$used.memory.centered < 100)), ]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (disabling sparse matrices)"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 200 & bm.mboost$used.memory.centered >= 0, ]
bm.mboost = bm.mboost[! ((seq_along(bm.mboost$second) > 10) & (bm.mboost$used.memory.centered < 140)), ]


bm.algos.linear = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.bad[81:791, ]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (enabling sparse matrices)"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 2000 & bm.mboost$used.memory.centered >= 0, ]
bm.mboost = bm.mboost[! ((seq_along(bm.mboost$second) > 10) & (bm.mboost$used.memory.centered < 120)), ]

bm.algos.linear = rbind(bm.algos.linear, bm.mboost)





bm.file.good = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters7.txt",
  header = TRUE,
  sep = ";"
)
bm.file.good$used.memory.centered = bm.file.good$used.memory - bm.file.good$used.memory[1]
bm.file.good$second = bm.file.good$second / 60

bm.file.bad = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters8.txt",
  header = TRUE,
  sep = ";"
)
bm.file.bad$used.memory.centered = bm.file.bad$used.memory - bm.file.bad$used.memory[1]
bm.file.bad$second = bm.file.bad$second / 60

bm.compboost = bm.file.good[8:489, ]
bm.mboost    = bm.file.good[490:1920, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"
bm.compboost = bm.compboost[bm.compboost$used.memory.centered < 420 & bm.compboost$used.memory.centered > 0, ]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (disabling sparse matrices)"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 788 & bm.mboost$used.memory.centered >= 0, ]

bm.algos.spline = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.bad[513:2011, ]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1] - 200
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (enabling sparse matrices)"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 220 & bm.mboost$used.memory.centered >= 0, ]

bm.algos.spline = rbind(bm.algos.spline, bm.mboost)



plotMemResults(bm.algos.linear, bm.algos.spline, "Memory Comparison using 5000 Iterations",
  "Elapsed Minutes", 0, 800)
