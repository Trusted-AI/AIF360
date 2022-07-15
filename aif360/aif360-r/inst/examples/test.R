library(aif360)
install_aif360()


load_aif360_lib()



dd <- aif360::binary_label_dataset(
  data_path = system.file("extdata", "data.csv", package="aif360"),
  favor_label=0,
  unfavor_label=1,
  unprivileged_protected_attribute=0,
  privileged_protected_attribute=1,
  target_column="income",
  protected_attribute="sex")

dd$favorable_label
dd$labels
dd$unfavorable_label
