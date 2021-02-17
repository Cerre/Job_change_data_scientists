import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# ML classifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestClassifier

# scaling
from sklearn.preprocessing import StandardScaler

# pipeline
from imblearn.over_sampling import SMOTE 

# pipe
from sklearn.pipeline import Pipeline

# ML classifier model Evaluation
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer





def read_data():
    train_file = 'data/aug_train.csv'
    test_file = 'data/aug_test.csv'

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    train_headers = list(train_data.columns)
    return train_data, test_data, train_headers



def replace_nan_values(df):
    df.fillna({'gender': 'Other'}, inplace = True)
    df.fillna({'enrolled_university': 'no_enrollment'}, inplace = True)
    df.fillna({'education_level': 'Graduate'}, inplace = True)
    df.fillna({'major_discipline': 'Other'}, inplace = True)



def early_checks(train_df):
    # print('Number of duplicated rows: {}\n'.format(sum(train_df.duplicated())))
    print('Columns with NaN values\n {}\n'.format(train_df.isna().any()))
    # print('Rows with 50% or more missing values {}\n'.format([x for x in enumerate(train_df.isnull().sum(axis = 1)) if x[1] >= (len(list(train_df.columns)))/2]))



def categorize_labels(df):
    df['enrolled_university'] = df['enrolled_university'].map({'no_enrollment': 0, 'Full time course': 2, 'Part time course': 1})
    df['education_level'] = df['education_level'].map({'Primary School': 0, 'High School': 1, 'Graduate': 2, 'Masters': 3, 'Phd': 3})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': -1, 'Other': 0})
    df['major_discipline'] = df['major_discipline'].map({'STEM': 5, 'Business Degree': 4, 'Humanities': 3, 'Arts': 2, 'No Major': 1, 'Other': 0})

    df['experience'].replace({'>20': 21, '<1': 0}, inplace = True)
    df['experience'] = df['experience'].apply(lambda x : int(x) if pd.notnull(x) else x)

    df['company_size'] = df['company_size'].map({'50-99': 2, '<10': 0, '10/49': 1, '100-500': 3, '500-999': 4, '1000-4999': 5, '5000-9999': 6, '10000+': 7})

    df['company_type'] = df['company_type'].map({'Pvt Ltd': 0, 'Funded Startup': 1, 'Early Stage Startup': 2, 'Other': 3, 'Public Sector': 4, 'NGO': 5})

    df['last_new_job'] = df['last_new_job'].map({'>4': 5, '1': 1, '4': 4, '3': 3, '2': 2, 'never': 0})

    df['city'] = pd.Categorical(df['city']).codes
    df['relevent_experience'] = pd.Categorical(df['relevent_experience']).codes
    return df

def plot_missing_values(df):
    # Missing value
    missing_value = 100 * df.isnull().sum()/len(df)
    missing_value = missing_value.reset_index()
    missing_value.columns = ['variables','missing values in percentage']
    fig = px.bar(missing_value, y='missing values in percentage',x='variables',title='Missing values % in each column',
             template='ggplot2');
    fig.show()

def check_unique_features(df, df2):
    for feature in df.columns:
        print('*******','Column name:',feature,'*******')
        print(df[feature].unique())
        print(df2[feature].unique())
        print('***********-end-***********')
        print(' ')

def fill_in_null_with_mice(df):
    mice_imputer = IterativeImputer()
    cols_with_nan = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']
    for key in cols_with_nan:
        df[key] = mice_imputer.fit_transform(df[[key]])
        df[key] = round(df[key])
    return df

def create_testsets(train, test):
    X_train = train[['city', 'city_development_index', 'gender',
       'relevent_experience', 'enrolled_university', 'education_level',
       'major_discipline', 'experience', 'company_size', 'company_type',
       'last_new_job', 'training_hours']]


    train_labels = train['target']

    X_test = test[[ 'city', 'city_development_index', 'gender',
       'relevent_experience', 'enrolled_university', 'education_level',
       'major_discipline', 'experience', 'company_size', 'company_type',
       'last_new_job', 'training_hours']]

    test_labels = np.load('data/jobchange_test_target_values.npy')
    # print(X_test)

    return X_train, train_labels, X_test, test_labels

def train_model(X_train, y_train):
    rf_pipe = Pipeline(steps =[ ('std_scale',StandardScaler()), ("RF",RandomForestClassifier(random_state=0,max_depth= 10, 
    max_features= 5,min_samples_leaf= 30, min_samples_split= 50, n_estimators= 500))])
    rf_pipe.fit(X_train,y_train)
    return rf_pipe
def main():
    train_df, test_df, train_headers = read_data()
    train_mice = train_df.copy()
    train_mice = categorize_labels(train_mice)
    # check_unique_features(train_mice, train_df)
    train_mice = fill_in_null_with_mice(train_mice)
    
    
    test_mice = test_df.copy()
    test_mice = categorize_labels(test_mice)
    test_mice = fill_in_null_with_mice(test_mice)

    X_train, y_train, X_test, y_test = create_testsets(train_mice, test_mice)
    rf_pipe = train_model(X_train, y_train)
    y_pred = rf_pipe.predict(X_train)
    print(classification_report(y_train, y_pred))

    
    
    








if __name__ == '__main__':
    main()

