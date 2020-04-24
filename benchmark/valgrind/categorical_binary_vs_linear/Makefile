all:
	make valgrind-linear
	make valgrind-binary

valgrind-linear:
	R -d "valgrind --tool=massif --stacks=yes  --verbose --trace-children=yes --massif-out-file=massif.out.cboost-linear-stack --log-file=log-cboost-linear-stack.txt" -e "source('categorical_linear.R')"
	R -d "valgrind --tool=massif --pages-as-heap=yes --verbose --trace-children=yes --massif-out-file=massif.out.cboost-linear-heap --log-file=log-cboost-linear-heap.txt" -e "source('categorical_linear.R')"

valgrind-binary:
	R -d "valgrind --tool=massif --stacks=yes  --verbose --trace-children=yes --massif-out-file=massif.out.cboost-binary-stack --log-file=log-cboost-binary-stack.txt" -e "source('categorical_binary.R')"
	R -d "valgrind --tool=massif --pages-as-heap=yes --verbose --trace-children=yes --massif-out-file=massif.out.cboost-binary-heap --log-file=log-cboost-binary-heap.txt" -e "source('categorical_binary.R')"


