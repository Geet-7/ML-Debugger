global issues 
issues = []
from utils.analysis import class_coverage
def max_class(df): 
    arr_class_im = class_coverage(df)
    max_class_val = arr_class_im.max()
    id_max_class_val = arr_class_im.idxmax()
    global flag_high_imb
    flag_high_imb = 0
    if(max_class_val >= 85):
        flag_high_imb = 1
        issues.append(("Fix class imbalance", "HIGH"))
        return(f"⚠️ Class {id_max_class_val} detected responsible for High Class Imbalance.       \nThe model may be biased toward the majority class and performance metrics like accuracy could be misleading ")
    if max_class_val >= 70 and max_class_val < 85:
        issues.append(("Fix class imbalance", "MEDIUM"))
        flag_high_imb = 2
        return(f"⚠️ Class {id_max_class_val} detected responsible for Moderate Class Imbalance.  \nThe model may show some bias toward the majority class ")
    else:
        flag_high_imb = 0
        return("No imbalance detected")



from utils.analysis import missing_percent
def missing_check(df): 
    arr_miss_per = missing_percent(df)
    max_miss_per_val = arr_miss_per.max()
    id_max_miss_per_val = arr_miss_per.idxmax()
    
    
    if max_miss_per_val > 30:
        issues.append((f"Handle missing values in Column {id_max_miss_per_val}", "HIGH"))
        return f"⚠️ Column {id_max_miss_per_val} has high missing data ({max_miss_per_val:.2f}%).  \nThis is a serious issue and should be handled carefully."
        
    
    elif 10 <= max_miss_per_val <= 30:
        issues.append((f"Handle missing values in Column {id_max_miss_per_val} ", "MEDIUM"))
        return f"⚠️ Column {id_max_miss_per_val} has moderate missing data ({max_miss_per_val:.2f}%).  \nConsider applying imputation techniques."
    
    
#the following is to find classes with very hih correlation and flag them 
def high_corr(df):
    df_numonly = df.select_dtypes(include='number')
    corr_mat = df_numonly.corr()
    
    high_corr_arr = []
    cols = corr_mat.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_mat.iloc[i, j]
            
            if abs(val) >= 0.9:
                high_corr_arr.append([cols[i], cols[j], val])
                issues.append(("Remove correlated features", "LOW"))
    return high_corr_arr


def high_corr_flag():
    from utils.issues import max_class
    if flag_high_imb == 1:
        return True

def issues_list():   
    priority_map = {
        "HIGH":3,
        "MEDIUM":2,
        "LOW":1
    }

    issues.sort(key = lambda x: priority_map[x[1]], reverse = True)
    return issues