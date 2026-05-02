
#basic info
def basic_info(df):

    return{
        "rows" : df.shape[0],
        "columns" : df.shape[1],
        "Column_list" : list(df.columns),
        "dtypes" : df.dtypes.astype(str)
    }
    
#missing percent 
#we will here calculate the percentage of missing values of the dataset
def missing_percent(df):
    missing_per_arr = df.isnull().mean() * 100
    return missing_per_arr.sort_values(ascending = False)


#finding imbalance
#the following function will calculate and store the percentage of coverage of each class
def class_coverage(df):
    class_coverage_arr = df.iloc[:, -1].value_counts(normalize=True)*100
    return class_coverage_arr.sort_values(ascending = False)

#Correlation
#the follwoing code is finding the correlation matrix of all the numeric variables
def correlation_matrix(df):
    df_numonly = df.select_dtypes(include = 'number')
    corr_matrix = df_numonly.corr()
    return(df_numonly.corr())