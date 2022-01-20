# FairAdapt called from R
wrapper <- function(train_data, test_data, adj_mat, res_vars = NULL, prot_attr,
    outcome) {

    prot_attr <- gsub("-", ".", prot_attr)
    outcome <- gsub("-", ".", outcome)

    train_data <- as.data.frame(
        lapply(train_data, function(x) {
            if (is.ordered(x)) class(x) <- "factor"
            x
        })
    )

    test_data <- as.data.frame(
        lapply(test_data, function(x) {
            if (is.ordered(x)) class(x) <- "factor"
            x
        })
    )

    adj.mat <- as.matrix(adj_mat)
    rownames(adj.mat) <- colnames(adj.mat) <- names(train_data)

    formula_adult <- as.formula(paste(outcome, "~ ."))
    L <- fairadapt::fairadapt(
        formula = formula_adult,
        train.data = train_data,
        test.data = test_data,
        adj.mat = adj.mat,
        prot.attr = prot_attr,
        res.vars = res_vars
    )

    names(L) <- c("train", "test")
    return(L)
}