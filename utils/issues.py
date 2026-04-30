from utils.analysis import class_coverage
def max_class(df): 
    arr_class_im = class_coverage(df)
    max_class_val = arr_class_im.max()
    id_max_class_val = arr_class_im.idxmax()
    
    if(max_class_val >= 85):
        return(f"⚠️ Class {id_max_class_val} detected responsible for High Class Imbalance.       \nThe model may be biased toward the majority class and performance metrics like accuracy could be misleading ")
    
    if max_class_val >= 70 and max_class_val < 85:
        return(f"⚠️ Class {id_max_class_val} detected responsible for Moderate Class Imbalance.  \nThe model may show some bias toward the majority class ")
    else:
        return("No imbalance detected")



from utils.analysis import missing_percent
def max_class(df): 
    arr_miss_per = missing_percent(df)
    max_miss_per_val = arr_miss_per.max()
    id_max_miss_per_val = arr_miss_per.idxmax()
    
    
    if max_miss_per_val > 30:
        return f"⚠️ Column {id_max_miss_per_val} has high missing data ({max_miss_per_val:.2f}%).  \nThis is a serious issue and should be handled carefully."
    
    elif 10 <= max_miss_per_val <= 30:
        return f"⚠️ Column {id_max_miss_per_val} has moderate missing data ({max_miss_per_val:.2f}%).  \nConsider applying imputation techniques."
    
    else:
        return f"Column {id_max_miss_per_val} has low missing data ({max_miss_per_val:.2f}%).  \nNo major issue."