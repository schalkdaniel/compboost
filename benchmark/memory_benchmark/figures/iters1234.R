bm.file.good = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters1.txt",
  header = TRUE,
  sep = ";"
)
bm.file.good$used.memory.centered = bm.file.good$used.memory - bm.file.good$used.memory[1]
bm.file.good$second = bm.file.good$second / 60

bm.file.bad = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters2.txt",
  header = TRUE,
  sep = ";"
)
bm.file.bad$used.memory.centered = bm.file.bad$used.memory - bm.file.bad$used.memory[1]
bm.file.bad$second = bm.file.bad$second / 60

bm.compboost = bm.file.good[7:25, ]
bm.mboost    = bm.file.good[26:229, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (disabling sparse matrices)"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 250 & bm.mboost$used.memory.centered > 0, ]

bm.algos.good = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.bad[24:229, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (enabling sparse matrices)"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 250 & bm.mboost$used.memory.centered > 0, ]

bm.algos.linear = rbind(bm.algos.good, bm.mboost)






bm.file.good = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters3.txt",
  header = TRUE,
  sep = ";"
)
bm.file.good$used.memory.centered = bm.file.good$used.memory - bm.file.good$used.memory[1]
bm.file.good$second = bm.file.good$second / 60

bm.file.bad = read.table(
  file = "R/mem_benchmark/memory_track/bm_mem_iters4.txt",
  header = TRUE,
  sep = ";"
)
bm.file.bad$used.memory.centered = bm.file.bad$used.memory - bm.file.bad$used.memory[1]
bm.file.bad$second = bm.file.bad$second / 60

bm.compboost = bm.file.good[5:104, ]
bm.mboost    = bm.file.good[105:400, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]
bm.compboost$algo = "compboost"
bm.compboost = bm.compboost[bm.compboost$used.memory.centered < 310 & bm.compboost$used.memory.centered > 0, ]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (disabling sparse matrices)"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 400 & bm.mboost$used.memory.centered > 0, ]

bm.algos.good = rbind(bm.compboost, bm.mboost)

bm.mboost = bm.file.bad[112:435, ]

bm.compboost$used.memory.centered = bm.compboost$used.memory.centered - bm.compboost$used.memory.centered[1]
bm.compboost$second = bm.compboost$second - bm.compboost$second[1]

bm.mboost$used.memory.centered = bm.mboost$used.memory.centered - bm.mboost$used.memory.centered[1]
bm.mboost$second = bm.mboost$second - bm.mboost$second[1]
bm.mboost$algo = "mboost (enabling sparse matrices)"
bm.mboost    = bm.mboost[bm.mboost$used.memory.centered < 200 & bm.mboost$used.memory.centered > 0, ]

bm.algos.spline = rbind(bm.algos.good, bm.mboost)




plotMemResults(bm.algos.linear, bm.algos.spline, "Base-Line Memory Comparison",
  "Elapsed Seconds", 0, 400)
