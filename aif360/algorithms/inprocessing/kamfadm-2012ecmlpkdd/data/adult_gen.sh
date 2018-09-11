#!/usr/bin/env bash
# generated discritized adult.data
# INPUT: adult.test in adult@UCI Repository
# OUTPUT: adultd.arff, adultd.data adult.bindata

w=${HOME}/work/python
o=00ORIG
d=00DATA

tmp=chotto-$$
stem=adult

echo "convert to arff format"
cat $o/${stem}.data $o/${stem}.test | \
  python adult_arff.py -o $d/${stem}.arff

echo "convert to the space separated format"
python $w/arff2txt.py -m 1 $d/${stem}.arff $d/${tmp}.data
python $w/arff2txt.py -m 3 $d/${stem}.arff $d/${tmp}.bindata

echo "move the sentitive attribute to the last position"
python select_sensitive.py -n -f 9  $d/${tmp}.data    $d/${stem}.data
python select_sensitive.py -n -f 59 $d/${tmp}.bindata $d/${stem}.bindata

echo "clearn-up tmporary files"
/bin/rm -f $d/${tmp}*
