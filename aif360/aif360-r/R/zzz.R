## quiets concerns of R CMD check re: the .'s that appear in pipelines
if(getRversion() >= "2.15.1")
    utils::globalVariables(c("datasets", "metrics", "tf", "pre_algo", "in_algo", "post_algo"))
Globals <- list()
