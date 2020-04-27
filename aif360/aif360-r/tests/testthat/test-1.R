# load aif library
library(reticulate)
install_aif360(method='virtualenv', envname='check-pack', conda_python_version = '3.7')
reticulate::use_virtualenv('check-pack')
load_aif360_lib()
