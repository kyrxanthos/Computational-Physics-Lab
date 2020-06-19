# this makefile runs my 2 python programs.
# run2 has a dependency on the first program which creates the lists 
# and then it uses the lists to produce the plots

x_y.png z_t.png : Ph20_A2.1.py t_export.csv x_export.csv y_export.csv z_export.csv
	python3 Ph20_A2.1.py 12

%.csv : Ph20_A2.py
	python3 Ph20_A2.py 1 1 1 1.1 0 0.01 1000

.PHONY : clean
clean : 
	rm -f *.csv 
	rm -f *.png




