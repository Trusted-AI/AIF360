#!/usr/bin/env bash
# generated discritized adult.data
# INPUT: adult.test in adult@UCI Repository
# OUTPUT: adultd.arff, adultd.data adult.bindata

w=${HOME}/work/python
o=00ORIG
d=00DATA

tmp=chotto-$$
input=adult.test

echo "convert to arff format"
python adult_arff.py $o/${input} $d/${tmp}.arff

echo "discritize by the Calders and Verwer's procedure"
python adult_discritize.py $d/${tmp}.arff $d/adultd.arff

echo "convert to the space separated format"
python $w/arff2txt.py -m 1 $d/adultd.arff $d/${tmp}.data
python $w/arff2txt.py -m 3 $d/adultd.arff $d/${tmp}.bindata

echo "move the sentitive attribute to the last position"
python select_sensitive.py -n -r -f 9 $d/${tmp}.data $d/adultd.data
python select_sensitive.py -n -r -f 67 $d/${tmp}.bindata $d/adultd.bindata

echo "generate data with quadratic terms"
python add_quad_terms.py -l 2 -i $d/adultd.bindata -o $d/adultdq1.bindata
python add_quad_terms.py -l 1 -i $d/adultd.bindata  | \
    python select_sensitive.py -f 80 | \
    cut -f 1-160,162-3403 -d\  > $d/adultdq2.bindata

echo "clearn-up tmporary files"
/bin/rm -f $d/${tmp}*
