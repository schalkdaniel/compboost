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
		suppressMessages({
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
    })
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
	runtime.plot.iterations = runtime.data.iterations %>% plotRuntimeBenchmark(header = "", xlab = "Number of Iterations"),
	runtime.plot.ncols = runtime.data.ncols %>% plotRuntimeBenchmark(header = "", xlab = "Number of Base-Learner"),	
  runtime.plot.nrows = runtime.data.nrows %>% plotRuntimeBenchmark(header = "", xlab = "Number of Observations"),


  # Raw data from the runtime benchmark:
  # --------------------------------------------------------
  # Measurements are in KiB, it is more convenient to use a list here:
  raw.memory.benchmark.data = {

    files = c("cboost_linear.rds", "cboost_spline.rds", "glmboost.rds", "gamboost.rds", "mboost_linear.rds", "mboost_spline.rds")

    list.df = list()
    for (file in files) {
      full.file = paste0("memory/benchmark_files/", file)
      if (file.exists(full.file)) {
        nm = load(full.file)
        list.df[[file]] = get(nm)
        if (file == "cboost_linear.rds") {
          algo = "compboost"
          learner = "linear"
        }        
        if (file == "cboost_spline.rds") {
          algo = "compboost"
          learner = "spline"
        }
        if (file == "glmboost.rds") {
          algo = "glmboost"
          learner = "linear"
        }
        if (file == "gamboost.rds") {
          algo = "gamboost"
          learner = "spline"
        }
        if (file == "mboost_linear.rds") {
          algo = "mboost"
          learner = "linear"
        }
        if (file == "mboost_spline.rds") {
          algo = "mboost"
          learner = "spline"
        }
        list.df[[file]]$algo = algo
        list.df[[file]]$learner = learner
      }
    }
    dplyr::bind_rows(list.df)
  },

	# Plot of the results of the memory benchmark:
	# --------------------------------------------------------

  memory.plot = raw.memory.benchmark.data %>% plotMemResults(mytitle = "", myxlab = "Elapsed Minutes"),

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