#!/usr/bin/env bash
# generate all artificial data

datadir=00DATA

# nos of samples
n=10000

# nos of features
f=20

# filename base
stembase=sdata_cv

# script name
script=sdata_cv.py

# random seed
rseed=1234

###### generate one set ######

function gen()
{
  stem=${datadir}/${stembase}-lb=${lb}-sb=${sb}
  base=${stem}@${t}.data

python $script --rseed=`expr $t + $rseed` \
    -n `expr $n \* 2` -l $lb -s $sb -f $f -o $base

  out=${stem}-c=lat@${t}l.data
  egrep -v '^#' $base | head -${n} | \
    cut -d\  -f 1-`expr $f + 1`,`expr $f + 2` > $out

  out=${stem}-c=obs@${t}l.data
  egrep -v '^#' $base | head -${n} | \
    cut -d\  -f 1-`expr $f + 1`,`expr $f + 3` > $out

  out=${stem}-c=lat@${t}t.data
  egrep -v '^#' $base | tail -${n} | \
    cut -d\  -f 1-`expr $f + 1`,`expr $f + 2` > $out

  out=${stem}-c=obs@${t}t.data
  egrep -v '^#' $base | tail -${n} | \
    cut -d\  -f 1-`expr $f + 1`,`expr $f + 3` > $out
}

###### main generator ######

# init
if [ ! -d ${datadir} ]; then
  mkdir -p $datadir
fi

# main loop
for t in 0 1 2 3 4; do
  for lb in 0.2 0.4 0.8; do
    for sb in 0.2 0.4 0.8; do
      gen
    done
  done
done
