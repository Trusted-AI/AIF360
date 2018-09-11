Data Format
===========

Format of each line::

  <X1><sp><X2><sp>...<Xk><sp><S><sp><Y><nl>

Xn
  values of the n-th non-sensitive feature
S
  values of a sensitive feature. 0=protected group, 1=non-protected group
Y
  values of a target variable. 0=not preferred class, 1=preferred class,

UCI Repository
==============

adult
-----

* adult / Census income 
* sensitive status: 10th "sex" is female
* nfv = 0:8:0:16:0:7:14:6:5:0:0:0:41:(2)

adultd
------

* Discritized and small sized version of adult
* sensitive status: 10th "sex" is female
* nfv = 4:7:4:16:4:7:14:6:5:2:2:3:8:(2)

creditg_p
---------

* credit-g 
* sensitive-status: personal_status=female div/dep/mar (female:
  divorced/separated/married)
* nfv = 4:0:5:11:0:5:5:0:3:0:4:0:3:3:0:4:0:2:2:(2)

creditg_j
---------

* credit-g
* sensitive-status: job=unemp/unskilled non res (unemployed/ unskilled -
  non-resident)
* nfv = 4:0:5:11:0:5:5:0:5:3:0:4:0:3:3:0:0:2:2:(2)

credit_f
--------

* credit-g
* sensitive-status: forign_worker=yes
* nfv = 4:0:5:11:0:5:5:0:5:3:0:4:0:3:3:0:4:0:2:(2)



Synthetic Data Sets
===================

sdata_cv
--------

- synthetic data set in Caldars and Verwer DMKD2010
