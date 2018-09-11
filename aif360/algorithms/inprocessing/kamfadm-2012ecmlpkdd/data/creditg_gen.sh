#!/usr/bin/env bash
# generated discritized adult.data
# INPUT: adult.test in adult@UCI Repository
# OUTPUT: adultd.arff, adultd.data adult.bindata

w=${HOME}/work/python
o=00ORIG
d=00DATA

#tmp=chotto-$$
in=credit-g
out=creditg

echo "convert to the space separated format"
python $w/arff2txt.py -m 1 $o/${in}.arff $d/${out}.data
python $w/arff2txt.py -m 3 $o/${in}.arff $d/${out}.bindata

echo "move the sentitive attribute to the last position"
python creditg_p_data.py    < $d/${out}.data > $d/${out}_p.data
python creditg_j_data.py    < $d/${out}.data > $d/${out}_j.data
python creditg_f_data.py    < $d/${out}.data > $d/${out}_f.data
python creditg_p_bindata.py < $d/${out}.data > $d/${out}_p.bindata
python creditg_j_bindata.py < $d/${out}.data > $d/${out}_j.bindata
python creditg_f_bindata.py < $d/${out}.data > $d/${out}_f.bindata
