# =================================================================================================== #
#                                                                                                     #
#                                     Create Readme Using Drake                                       #
#                                                                                                     #
# =================================================================================================== #

library(drake)
library(batchtools)
library(dplyr)
library(tidyr)
library(gtable)
library(ggplot2)
library(grid)
library(gridExtra)

## Source plot functions:
## ==============================================

source("benchmark_plot_functions.R")

## Create drake plan:
## ==============================================

benchmark.plan = drake_plan(

	# Used system:
	# --------------------------------------------------------
	my.system = {
		spec = Sys.info()["sysname"]
		out.string = paste0("This benchmark was executed on a `", spec, "` machine ")

    if (spec == "Linux") {
    	pc.info = strsplit(system(command = "lsb_release -a | grep Description:*", intern = TRUE), split = "\\t")[[1]][2]
    	
    	processor = strsplit(system(command = "lshw -short | grep processor:*", intern = TRUE), split = "processor")[[1]][2]
    	processor = substr(processor, start = 7, stop = nchar(processor))
    
    	memory = strsplit(system(command = "lshw -short | grep memory:*", intern = TRUE), split = "memory")[[1]][2]
    	memory = substr(memory, start = 10, stop = nchar(memory))

    	out.string = paste0(out.string, "with a `", processor, " ", memory, "` ")
    }
    out.string = paste0(out.string, "using the `R` package `batchtools`.")
    out.string
	},

	# Raw data from the runtime benchmark:
	# --------------------------------------------------------
	raw.runtime.benchmark.data = {
		runtime.registry = loadRegistry("runtime/benchmark_files")
    dt.bm.runtime = unwrap(reduceResultsDataTable(ids = findDone(), reg = runtime.registry))

    # Time in Minutes:
    dt.bm.runtime$time  = sapply(dt.bm.runtime$time, function(x) as.numeric(x)[3]) / 60
    dt.bm.runtime$nrows = sapply(dt.bm.runtime$data.dim, function(x) as.numeric(x)[1])
    dt.bm.runtime$ncols = sapply(dt.bm.runtime$data.dim, function(x) as.numeric(x)[2])
    dt.bm.runtime$data.dim = NULL
    
    if (nrow(getJobPars(ids = findNotDone(reg = runtime.registry))) > 0) {
      # Bind non finished jobs with a time of 0:
      dt.bm.runtime %>% bind_rows(
        unwrap(getJobPars(ids = findNotDone(reg = runtime.registry))) %>%
          mutate(algo = ifelse(algorithm == "mboost", "mboost", 
          	ifelse(algorithm == "compboost", "compboost",
          		ifelse((algorithm == "mboost.fast") & (learner == "linear"), "glmboost", "gamboost")
            )),
            time = 0, p = p + 1) %>%
          select(-problem) %>%
          rename(nrows = n, ncols = p, learner = algorithm)
        ) %>%
        arrange(job.id)
    } else {
    	dt.bm.runtime
    }
	},
	# Preprocessing of raw runtime data:
	# --------------------------------------------------------
	runtime.data.iterations = raw.runtime.benchmark.data %>%
    filter(job.id <= 210) %>%
    group_by(learner, iters, algo) %>%
    summarize(x.value = iters[1], Time = median(time), Algorithm = algo[1],
      Time.min = min(time), Time.max = max(time)) %>%
    mutate(rel.factor = ifelse(algo == "compboost", 1, 
    	ifelse(algo == "mboost", Time[algo == "mboost"] / Time[algo == "compboost"], 
    		Time[algo %in% c("gamboost", "glmboost")] / Time[algo == "compboost"])
    )),
  runtime.data.ncols = raw.runtime.benchmark.data %>%
    filter((job.id > 210 & job.id <= 280) | (job.id > 350 & job.id <= 420) | (job.id > 490 & job.id <= 560)) %>%
    group_by(learner, ncols, algo) %>%
    summarize(x.value = ncols[1] - 1, Time = median(time), Algorithm = algo[1],
      Time.min = min(time), Time.max = max(time)) %>%
    mutate(rel.factor = ifelse(algo == "compboost", 1, 
    	ifelse(algo == "mboost", Time[algo == "mboost"] / Time[algo == "compboost"], 
    		Time[algo %in% c("gamboost", "glmboost")] / Time[algo == "compboost"])
    )),
  runtime.data.nrows = raw.runtime.benchmark.data %>%
      filter((job.id > 280 & job.id <= 350) | (job.id > 420 & job.id <= 490) | (job.id > 560)) %>%
      group_by(learner, nrows, algo) %>%
      summarize(x.value = nrows[1], Time = median(time), Algorithm = algo[1],
        Time.min = min(time), Time.max = max(time)) %>%
      mutate(rel.factor = ifelse(algo == "compboost", 1, 
      	ifelse(algo == "mboost", Time[algo == "mboost"] / Time[algo == "compboost"], 
      		Time[algo %in% c("gamboost", "glmboost")] / Time[algo == "compboost"])
      )),
 	# Plots of the results of the runtime benchmark:
	# --------------------------------------------------------
	runtime.plot.iterations = runtime.data.iterations %>%
	  # plotRuntimeBenchmark(header = "Benchmark for Increasing Number of Iterations", xlab = "Number of Iterations"),
	  plotRuntimeBenchmark(header = "", xlab = "Number of Iterations"),

	runtime.plot.ncols = runtime.data.ncols %>%
	  plotRuntimeBenchmark(header = "", xlab = "Number of Base-Learner"),	

  runtime.plot.nrows = runtime.data.nrows %>%
    plotRuntimeBenchmark(header = "", xlab = "Number of Observations"),


  # Raw data from the runtime benchmark:
  # --------------------------------------------------------
  # Measurements are in KiB, it is more convenient to use a list here:
  # raw.memory.benchmark.data = {
  #   memory.registry = loadRegistry("memory/benchmark_files")
  #   list.bm.memory = reduceResultsList(ids = findDone(), reg = memory.registry)
  #   list.bm.memory = lapply(list.bm.memory, function (ll) {

  #     # Allocated memory in MB:
  #     ll$memory.diff = (ll$memory.after[["used"]] - ll$memory.before[["used"]]) / 1024
      
  #     ll$memory.after = NULL
  #     ll$memory.before = NULL

  #     # Elapsed time in minutes:
  #     ll$ellapsed.time = ll$time[["elapsed"]] / 60
  #     ll$time = NULL

  #     ll$nrows = ll$data.dim[1]
  #     ll$ncols = ll$data.dim[2]
  #     ll$data.dim = NULL

  #     return (ll)
  #   })
  #   dt.bm.memory = dplyr::bind_rows(list.bm.memory) %>%
  #     mutate(job.id = findDone(reg = memory.registry)[["job.id"]])
    
  #   if (nrow(getJobPars(ids = findNotDone(reg = memory.registry))) > 0) {
  #     # Bind non finished jobs with a memory diff of 0:
  #     dt.bm.memory %>% bind_rows(
  #       unwrap(getJobPars(ids = findNotDone(), reg = memory.registry)) %>%
  #         mutate(algo = ifelse(algorithm == "mboost", "mboost", 
  #           ifelse(algorithm == "compboost", "compboost",
  #             ifelse((algorithm == "mboost.fast") & (learner == "linear"), "glmboost", "gamboost")
  #           )),
  #           memory.diff = 0, ellapsed.time = 0, p = p + 1) %>%
  #         select(-problem) %>%
  #         rename(nrows = n, ncols = p, learner = algorithm)
  #       ) %>%
  #       arrange(job.id)
  #   } else {
  #     dt.bm.memory
  #   }
  # },

	# Plots of the results of the memory benchmark:
	# --------------------------------------------------------

	# Create report as Readme for GitHub:
	# --------------------------------------------------------

	readme = rmarkdown::render(
		knitr_in("Readme.Rmd"),
		output_file = file_out("Readme.md"),
		quiet = TRUE
	),

	# Make double quotes work without a warning:
	# --------------------------------------------------------
	strings_in_dots = "literals"
)


# Create results:
make(benchmark.plan)