import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, mode
import statistics
import random
from copy import deepcopy
import itertools

data = pd.read_csv('ML1/Tab.delimited.Cleaned.dataset.WITH.variable.labels.csv', sep = '\t', encoding = "ISO-8859-1")
data.head()

mixed_columns = ['flagsupplement1', 'flagsupplement2', 'flagsupplement3', 'iatexplicitart1', 'iatexplicitart2',
                'iatexplicitart3', 'iatexplicitart4', 'iatexplicitart5', 'iatexplicitart6', 'iatexplicitmath1',
                'iatexplicitmath2', 'iatexplicitmath3', 'iatexplicitmath4', 'iatexplicitmath5', 'iatexplicitmath6',
                'sysjust1', 'sysjust2', 'sysjust3', 'sysjust4', 'sysjust5', 'sysjust6', 'sysjust7', 'sysjust8',
                'mturk.non.US', 'exprace']

all_NAN_columns = ['task_status', 'task_sequence', 'beginlocaltime']

numeric_columns = ['anchoring3ameter', 'anchoring3bmeter', 'anchoring1akm', 'anchoring1bkm', 'priorexposure8',
                    'mturk.duplicate', 'mturk.exclude.null', 'mturk.keep', 'mturk.exclude', 'gamblerfallacya_sd',
                    'gamblerfallacyb_sd', 'totexpmissed', 'numparticipants_actual', 'session_id', 'IATfilter',
                    'priorexposure3', 'priorexposure4', 'priorexposure5', 'priorexposure6', 'priorexposure7', 
                    'priorexposure9', 'priorexposure10', 'priorexposure11', 'priorexposure12', 'priorexposure13',
                    'numparticipants', 'age', 'anchoring1a', 'priorexposure1', 'priorexposure2', 'mturk.total.mini.exps',
                    'anchoring1b', 'anchoring2a', 'anchoring2b', 'anchoring3a', 'anchoring3b', 'anchoring4a',
                    'anchoring4b', 'artwarm', 'd_donotuse', 'ethnicity', 'flagdv1', 'flagdv2', 'flagdv3',
                    'flagdv4', 'flagdv5', 'flagdv6', 'flagdv7', 'flagdv8', 'gamblerfallacya', 'gamblerfallacyb',
                    'imaginedexplicit1', 'imaginedexplicit2', 'imaginedexplicit3', 'imaginedexplicit4',
                    'major', 'mathwarm', 'moneyagea', 'moneyageb', 'moneygendera', 'moneygenderb', 'omdimc3rt',
                    'omdimc3trt', 'quotea', 'quoteb', 'sunkcosta', 'sunkcostb', 'user_id', 'previous_session_id',
                    'order', 'meanlatency', 'meanerror', 'block2_meanerror', 'block3_meanerror', 'block5_meanerror',
                    'block6_meanerror', 'lat11', 'lat12', 'lat21', 'lat22', 'sd1', 'sd2', 'd_art1', 'd_art2', 'd_art',
                    'sunkDV', 'anchoring1', 'anchoring2', 'anchoring3', 'anchoring4', 'Ranchori', 'RAN001',
                    'RAN002', 'RAN003', 'Ranch1', 'Ranch2', 'Ranch3', 'Ranch4', 'gambfalDV', 'scalesreca', 'scalesrecb',
                    'reciprocityother', 'quotearec', 'quotebrec', 'quote', 'totalflagestimations', 'totalnoflagtimeestimations',
                    'flagdv', 'Sysjust', 'moneyfilter', 'Imagineddv', 'IATexpart', 'IATexpmath', 'IATexp.overall',
                    'IATEXPfilter', 'scalesorder', 'reciprocorder', 'diseaseforder', 'quoteorder', 'flagprimorder',
                    'sunkcostorder', 'anchorinorder', 'allowedforder', 'gamblerforder', 'moneypriorder', 'imaginedorder',
                    'iatorder']

date_columns = [col for col in data.columns if '_date' in col]

non_string_columns = mixed_columns + all_NAN_columns + numeric_columns + date_columns
string_columns = sorted([col for col in data.columns if col not in non_string_columns])

data = data.replace({'': np.nan, ' ': np.nan, '.': np.nan})

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col])

for col in date_columns:
    data[col] = pd.to_datetime(data[col])

for col in string_columns:
    data[col] = data[col].astype(str)

data["flagsupplement1"] = data["flagsupplement1"].apply(lambda x:
                                                    '11' if x == 'Very much' else 
                                                    ('1' if x == 'Not at all' else x))
data["flagsupplement2"] = data["flagsupplement2"].apply(lambda x:
                                                    '1' if x == 'Democrat' else 
                                                    ('7' if x == 'Republican' else x))
data["flagsupplement3"] = data["flagsupplement3"].apply(lambda x:
                                                    '1' if x == 'Liberal' else 
                                                    ('7' if x == 'Conservative' else x))

data["iatexplicitart1"] = data["iatexplicitart1"].apply(lambda x:
                                                    '1' if x == 'Very bad' else 
                                                    ('2' if x == 'Moderately bad' else x))
data["iatexplicitart2"] = data["iatexplicitart2"].apply(lambda x:
                                                    '1' if x == 'Very Sad' else 
                                                    ('2' if x == 'Moderately Sad' else x))
data["iatexplicitart3"] = data["iatexplicitart3"].apply(lambda x:
                                                    '1' if x == 'Very Ugly' else 
                                                    ('2' if x == 'Moderately Ugly' else x))
data["iatexplicitart4"] = data["iatexplicitart4"].apply(lambda x:
                                                    '1' if x == 'Very Disgusting' else 
                                                    ('2' if x == 'Moderately Disgusting' else x))
data["iatexplicitart5"] = data["iatexplicitart5"].apply(lambda x:
                                                    '1' if x == 'Very Avoid' else 
                                                    ('2' if x == 'Moderately Avoid' else x))
data["iatexplicitart6"] = data["iatexplicitart6"].apply(lambda x:
                                                    '1' if x == 'Very Afraid' else 
                                                    ('2' if x == 'Moderately Afraid' else x))

data["iatexplicitmath1"] = data["iatexplicitmath1"].apply(lambda x:
                                                    '1' if x == 'Very bad' else 
                                                    ('2' if x == 'Moderately bad' else
                                                    ('3' if x == 'Slightly bad' else x)))
data["iatexplicitmath2"] = data["iatexplicitmath2"].apply(lambda x:
                                                    '1' if x == 'Very Sad' else 
                                                    ('2' if x == 'Moderately Sad' else
                                                    ('3' if x == 'Slightly Sad' else x)))
data["iatexplicitmath3"] = data["iatexplicitmath3"].apply(lambda x:
                                                    '1' if x == 'Very Ugly' else 
                                                    ('2' if x == 'Moderately Ugly' else
                                                    ('3' if x == 'Slightly Ugly' else x)))
data["iatexplicitmath4"] = data["iatexplicitmath4"].apply(lambda x:
                                                    '1' if x == 'Very Disgusting' else 
                                                    ('2' if x == 'Moderately Disgusting' else
                                                    ('3' if x == 'Slightly Disgusting' else x)))
data["iatexplicitmath5"] = data["iatexplicitmath5"].apply(lambda x:
                                                    '1' if x == 'Very Avoid' else 
                                                    ('2' if x == 'Moderately Avoid' else
                                                    ('3' if x == 'Slightly Avoid' else x)))
data["iatexplicitmath6"] = data["iatexplicitmath6"].apply(lambda x:
                                                    '1' if x == 'Very Afraid' else 
                                                    ('2' if x == 'Moderately Afraid' else
                                                    ('3' if x == 'Slightly Afraid' else x)))

for col in mixed_columns:
    if "sysjust" in col:
        data[col] = data[col].apply(lambda x:
                                    '1' if x == 'Strongly disagree' else 
                                    ('7' if x == 'Strongly agree' else x))

data["mturk.non.US"] = data["mturk.non.US"].apply(lambda x: '1' if x == 'non-US IP address' else x)
data["exprace"] = data["exprace"].apply(lambda x: '11' if x == 'brazilwhite' else 
                                        ('12' if x == 'brazilblack' else
                                        ('13' if x == 'brazilbrown' else
                                        ('14' if x == 'chinese' else
                                        ('15' if x == 'malay' else
                                        ('16' if x == 'dutch' else x))))))

class MeanMode():
    def __init__(self, numeric_columns):
        self.numeric_columns = numeric_columns
        
    def predict(self, feature_name, train):
        if feature_name in self.numeric_columns:
            return np.mean(train)
        else:
            return mode(train)
    
    def get_mse(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.sum((y_true - y_pred)**2)/y_true.shape[0]

mean_mode_model = MeanMode(numeric_columns=numeric_columns)

# class VAE():
#     # Sigmoid
#     def sigmoid(self, x, derivative=False):
#         sig = 1/(1+np.exp(-x))
#         return sig*(1-sig) if derivative else sig
    
#     # Relu
#     def relu(self, x, derivative=False):
#         return 1.0*(x > 0) if derivative else x*(x > 0)
    
#     # Leaky Relu
#     def leaky_relu(self, x, alpha=0.01, derivative=False):
#         if derivative:
#             return 1 if x > 0 else alpha
#         return x if x > 0 else alpha*x

class Pipeline():
    def __init__(self, data, mean_mode_model):
        self.data = data
        self.data = self.data.drop(columns=['task_sequence', 'task_status', 'beginlocaltime', 'session_creation_date', 'session_last_update_date'])
        self.data = self.data.replace({'': np.nan, ' ': np.nan, '.': np.nan})
        self.data["expcomments"] = self.data["expcomments"].replace({'nan': np.nan})
        self.mean_mode = mean_mode_model
    
    def count_missing(self, data):
        return data.isnull().sum()
    
    def missing_value_perc(self):
        missing_value_data = (self.data.isnull().sum()*100/len(self.data)).reset_index()
        missing_value_data.columns = ["feature", "perc"]
        full_value_data = missing_value_data[missing_value_data["perc"] == 0]
        missing_value_data = missing_value_data[missing_value_data["perc"] > 0]
        missing_value_data = missing_value_data.sort_values(by=['perc'])
        missing_value_data = missing_value_data.set_index("feature")
        for ind in missing_value_data.index:
            if "_date" in ind:
                missing_value_data = missing_value_data.drop(ind, axis=0)
        missing_value_data = missing_value_data.reset_index()
        return missing_value_data, full_value_data

    def create_full_data(self, data):
        data_without_nan = data.drop(pd.isnull(data).any(1).nonzero()[0])
        return data_without_nan

    def k_fold(self, indexes, fold_value=10):
        splits = []
        indexes_copy = list(indexes)
        fold_lengths = int(len(indexes)/fold_value)
        for i in range(fold_value):
            fold = indexes_copy[i*fold_lengths:i*fold_lengths+fold_lengths]
            splits.append(fold)

        if splits[-1][-1] < len(indexes_copy):
            for i in range(splits[-1][-1]+1, len(indexes_copy)):
                splits[-1].append(i)
                
        return splits
    
    def cross_validation(self, train_data, X_test, y_test, model, feature_name):
        preds = []
        losses = []
        
        indexes = list(train_data.index)
        k_fold_splits = self.k_fold(indexes, fold_value=10)
        print(len(indexes))
        if model == "mean_mode":
            for i in range(len(k_fold_splits)):
                k_copy = k_fold_splits.copy()
                del k_copy[i]
                val = k_fold_splits[i]
                train_i = list(itertools.chain.from_iterable(k_copy))
                print(max(val), max(train_i))
                val_data = train_data.iloc[val,:]
#                 new_train_data = train_data.iloc[train_i,:]
#                 X_train = new_train_data.drop(feature_name, axis=1)
#                 y_train = new_train_data[feature_name]
#                 X_val = val_data.drop(feature_name, axis=1)
#                 y_val = val_data[feature_name]
#                 y_pred = y_val*np.nan

#                 value = self.mean_mode.predict(feature_name, y_train)
#                 y_pred = y_pred.fillna(value)
#                 loss = self.mean_mode.get_mse(y_val, y_pred)
#                 preds.append(value)
#                 losses.append(loss)
        
#             predicted_mean = statistics.mean(preds)
#             y_test = y_test.fillna(predicted_mean)
                
        return 1, 1
#         return y_test, statistics.mean(losses)      
            
    def train_test_data(self, feature_name):
        train_data = self.data[self.full_value_cols + [feature_name]]
        test_data = train_data[train_data[feature_name].isnull()]
        train_data = self.create_full_data(train_data)
        return train_data, test_data

    def impute_to_main_data(self, new_data, feature_name):
        indexes = self.data[feature_name].index[self.data[feature_name].apply(np.isnan)]
        for index in indexes:
            self.data.ix[index, feature_name] = new_data.ix[index, feature_name]
            # print(self.count_missing(self.data[feature_name]))
    
    def subset_data_based_on_missing_percentage(self):
        missing_value_data, full_value_data = self.missing_value_perc()
        self.full_value_cols = list(full_value_data['feature'])
        
        subsets = [10, 30, 50, 70, 100]
        lags = [0, 10, 30, 50, 70]
        for subset,lag in zip(subsets, lags):
            # Finding columns which have missing value less than a subset value
            missing_columns = list(missing_value_data[(missing_value_data["perc"] <= subset) & (missing_value_data["perc"] > lag)].feature)
            if 'task_creation_date.45' in missing_columns:
                missing_columns.remove('task_creation_date.45')
            print("########################################################################")
            print("Subset: {}\tNumber of missing value columns: {}".format(subset, len(missing_columns)))
            print("########################################################################")
            for feature_name in missing_columns:
                print("Feature Name: {}, Full columns: {}".format(feature_name, len(self.full_value_cols)))                
                train_data, test_data = self.train_test_data(feature_name)
                train_data = train_data.reset_index()
                X_train = train_data.drop(feature_name, axis=1)
                y_train = train_data[feature_name]
                X_test = test_data.drop(feature_name, axis=1)
                y_test = test_data[feature_name]
            
                for methods in ["mean_mode"]:
                    y_test, loss = self.cross_validation(train_data, X_test, y_test, methods, feature_name)
#                     print(loss)
#                     test_data = pd.concat([X_test, y_test], axis=1)
# #                     print(self.count_missing(test_data[feature_name]))
#                     train_data = train_data.append(test_data)
#                     self.impute_to_main_data(train_data, feature_name)
#                     self.full_value_cols.append(feature_name)
#                     print(self.count_missing(train_data[feature_name]))

pipeline = Pipeline(data, mean_mode_model)
pipeline.subset_data_based_on_missing_percentage()