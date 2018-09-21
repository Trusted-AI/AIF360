#!/usr/bin/env Rscript

# This R script can be used to download the Medical Expenditure Panel Survey (MEPS)
# data files for 2015 and 2016 and convert the files from SAS transport format into
# standard CSV files.

usage_note <- paste("",
    "By using this script you acknowledge the responsibility for reading and",
    "abiding by any copyright/usage rules and restrictions as stated on the",
    "MEPS web site (https://meps.ahrq.gov/data_stats/data_use.jsp).",
    "",
    "Continue [y/n]? > ", sep = "\n")

cat(usage_note)
answer <- scan("stdin", character(), n=1, quiet=TRUE)

if (tolower(answer) != 'y') {
    opt <- options(show.error.messages=FALSE)
    on.exit(options(opt))
    stop()
}

if (!require("foreign")) {
    install.packages("foreign")
    library(foreign)
}

convert <- function(ssp_file, csv_file) {
    message("Loading dataframe from file: ", ssp_file)
    df = read.xport(ssp_file)
    message("Exporting dataframe to file: ", csv_file)
    write.csv(df, file=csv_file, row.names=FALSE, quote=FALSE)
}

for (dataset in c("h181", "h192")) {
    zip_file <- paste(dataset, "ssp.zip", sep="")
    ssp_file <- paste(dataset, "ssp", sep=".")
    csv_file <- paste(dataset, "csv", sep=".")
    url <- paste("https://meps.ahrq.gov/mepsweb/data_files/pufs", zip_file, sep="/")

    # skip to next dataset if we already have the CSV file
    if (file.exists(csv_file)) {
        message(csv_file, " already exists")
        next
    }

    # download the zip file only if not downloaded before
    if (!file.exists(zip_file)) {
        download.file(url, destfile=zip_file)
    }

    # unzip and convert the dataset from SAS transport format to CSV
    unzip(zip_file)
    convert(ssp_file, csv_file)

    # clean up temporary files if we got the CSV file
    if (file.exists(csv_file)) {
        file.remove(zip_file)
        file.remove(ssp_file)
    }
}
