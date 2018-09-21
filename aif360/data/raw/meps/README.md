# Medical Expenditure Panel Survey data
<https://meps.ahrq.gov/mepsweb/>


The Medical Expenditure Panel Survey (MEPS) data consists of large scale surveys of families and individuals, medical providers, and employers, and collects data on health services used, costs & frequency of services, demographics, etc., of the respondents.

## Source / Data Set Description:


* [2015 full Year Consolidated Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): This file contains MEPS survey data for calendar year 2015 obtained in rounds 3, 4, and 5 of Panel 19, and rounds 1, 2, and 3 of Panel 20.

* [2016 full Year Consolidated Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192): : This file contains MEPS survey data for calendar year 2016 obtained in rounds 3, 4, and 5 of Panel 20, and rounds 1, 2, and 3 of Panel 21. 


## Data Use Agreement

As the user of the data it is your responsibility to read and abide by any copyright/usage rules and restrictions as
stated on the MEPS web site before downloading the data.

- [Data Use Agreement (2015 Data File)](https://meps.ahrq.gov/data_stats/download_data/pufs/h181/h181doc.shtml#Data)
- [Data Use Agreement (2016 Data File)](https://meps.ahrq.gov/data_stats/download_data/pufs/h192/h192doc.shtml#DataA)


## Download instructions

In order to use the MEPS datasets with AIF360, please follow the following directions to download the datafiles and 
convert into csv files. 

Follow either set of instructions below for using R or SPSS. Further instructions for SAS, and Stata, are available at 
the [AHRQ MEPS Github repository](https://github.com/HHS-AHRQ/MEPS).

 - **Generating CSV files with R**
    
    In the current folder run the R script `generate_data.R`. R can be downloaded from [CRAN](https://cran.r-project.org).
    If you are working on Mac OS X the easiest way to get the R command line support is by installing it with 
    [Homebrew](https://brew.sh/) `brew install R`.
    
    ```Bash
    cd aif360/data/raw/meps
    Rscript generate_data.R
    ```
    
    Example output:
    
    ```
    Loading required package: foreign
    
    trying URL 'https://meps.ahrq.gov/mepsweb/data_files/pufs/h181ssp.zip'
    Content type 'application/zip' length 13303652 bytes (12.7 MB)
    ==================================================
    downloaded 12.7 MB
    
    Loading dataframe from file: h181.ssp
    Exporting dataframe to file: h181.csv
    
    trying URL 'https://meps.ahrq.gov/mepsweb/data_files/pufs/h192ssp.zip'
    Content type 'application/zip' length 15505898 bytes (14.8 MB)
    ==================================================
    downloaded 14.8 MB
    
    Loading dataframe from file: h192.ssp
    Exporting dataframe to file: h192.csv
    ```

 - **Generating CSV files with SPSS**

    The instructions below require the use of SPSS.
    
    1. 2015 full Year Consolidated Data File
        * Download the [`Data File, ASCII format`](https://meps.ahrq.gov/mepsweb/data_files/pufs/h181dat.zip)
        * Extract the file `h181.dat` from downloaded zip archive
        * Convert the file to comma-delimited format, `h181.csv`, and save in this folder.
            * To convert the .dat file into csv format,download one of the programming statements files, such as the [SPSS Programming Statements](https://meps.ahrq.gov/mepsweb/data_stats/download_data/pufs/h181/h181spu.txt) file.
            * Edit this file to change the FILE HANDLE name to the complete path/name of the downloaded data file, execute the SPSS programming statements to load the data, and 'save as' a comma-delimited file called 'h181.csv' in the current folder.
        
    2. 2016 full Year Consolidated Data File
        * Download the [`Data File, ASCII format`](https://meps.ahrq.gov/mepsweb/data_files/pufs/h192dat.zip)
        * Extract the file `h192.dat` from downloaded zip archive
        * Convert the file to comma-delimited format, `h192.csv`, and save in current repository.
            * To convert the .dat file into csv format,download one of the programming statements files, such as the [SPSS Programming Statements](https://meps.ahrq.gov/mepsweb/data_stats/download_data/pufs/h192/h192spu.txt) file.
            * Edit this file to change the FILE HANDLE name to the complete path/name of the downloaded data file, execute the SPSS programming statements to load the data, and 'save as' a comma-delimited file called 'h192.csv' in this folder.