#!/bin/bash
for i in {1..32}
do
  wget https://raw.githubusercontent.com/pacslab/SLApp-PerfCost-MdlOpt/master/evaluations/alg/App6/perf_cost_data/App6_part$i.csv
done

