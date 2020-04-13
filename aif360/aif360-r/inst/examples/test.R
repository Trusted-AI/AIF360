library(raif360)
install_aif360()


load_aif360_lib()



dd <- raif360::aif_dataset(data_path='.... path to the pre_processed.csv input file',
                           favor_label=0,
                           unfavor_label=1,
                           unprivileged_protected_attribute=1,
                           privileged_protected_attribute=0,
                           target_column="Attrition",
                           protected_attribute="OverTime")
dd$favorable_label
dd$labels
dd$unfavorable_label
