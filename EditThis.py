# Import Dependencies
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
import csv
import codecs   
# Function to Load Train Dataset
def _load_train_dataset():
    training_data_path= 'training_data.csv'
    df_train = pd.read_csv(training_data_path)
    cols = df_train.columns
    cols = cols[:-2]
    train_features = df_train[cols]
    train_labels = df_train['prognosis']

    # Check for data sanity
    assert (len(train_features.iloc[0]) == 132)
    assert (len(train_labels) == train_features.shape[0])

    return train_features, train_labels, df_train

# Function to Load Test Dataset
def _load_test_dataset():
    test_data_path= 'Book1.csv'
    df_test = pd.read_csv(test_data_path)
    cols = df_test.columns
    cols = cols[:-1]
    test_features = df_test[cols]
    test_labels = df_test['prognosis']

    # Check for data sanity
    assert (len(test_features.iloc[0]) == 132)
    assert (len(test_labels) == test_features.shape[0])
        
    return test_features, test_labels, df_test

# Dataset Train Validation Split
def _train_val_split(train_features,train_labels):
    X_train, X_val, y_train, y_val = train_test_split(train_features,train_labels,
                                                      test_size=0.33,
                                                      random_state=101)
    return X_train, y_train, X_val, y_val

# ML Model
def train_model(train_features,train_labels,model_name,model_save_path):
    # Get the Data
    X_train, y_train, X_val, y_val = _train_val_split(train_features,train_labels)
    
    
    # Model Selection gridcv
    dec_tree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=6, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
    
#     ###
#     pipe = Pipeline(steps=[('std_slc', std_slc),
#                            ('pca', pca),
#                            ('dec_tree', dec_tree)])
#     n_components = list(range(1,train_features.shape[1]+1,1))
#     criterion = ['gini', 'entropy']
#     max_depth = [2,4,6,8,10,12]
#     parameters = dict(pca__n_components=n_components,
#                       dec_tree__criterion=criterion,
#                       dec_tree__max_depth=max_depth)
#     clf_GS = GridSearchCV(pipe, parameters)
#     clf_GS.fit(train_features, train_labels)
#     ###
    
    # Training the Model
    classifier = dec_tree.fit(X_train, y_train)
    # Trained Model Evaluation on Validation Dataset
    confidence = classifier.score(X_val, y_val)
    # Validation Data Prediction
    y_pred = classifier.predict(X_val)
    # Model Validation Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    score = cross_val_score(classifier, X_val, y_val, cv=3) 
        
#   # Save Trained Model
#   dump(classifier, str(model_save_path + model_name + ".joblib"))

    return classifier,score


# Function to Make Predictions on Test Data
def make_prediction(test_features,test_labels, our_model=None):
    clf=our_model
#     try:
#             # Load Trained Model
#             clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
#         except Exception as e:
#             print("Model not found...")

    if clf is None:
        print("Model Not found")
        return 0
    result = clf.predict(test_features)
        
    return result
if __name__ == "__main__":
    # Model Currently Training
    model_name = 'decision_tree'
    # Model Save Path
    model_save_path= './saved_model/'
    
    #Verbose for Log prints
    verbose= False
    
    # Load Training Data
    train_features,train_labels,train_df = _load_train_dataset()
    
    # Load Test Data
    test_features,test_labels,test_df = _load_test_dataset()
        
    # Training the model
    model,score=train_model(train_features,train_labels,model_name,model_save_path)
    #print("Cross validation Score: ", score)
    
    # Get Model Performance on Test Data
    ans=make_prediction(test_features=test_features, test_labels=test_labels, our_model=model)
    print(ans[0])
    
    in_file = open("Book1.csv", "r")   
    reader = csv.reader(in_file)
    my_list=list(reader)
    in_file.close()
    
    my_list[1][132]=ans[0]
    
    out_file = open("Book1.csv", "w", newline = '')
    csv_writer = csv.writer(out_file)
    csv_writer.writerows(my_list)    
    out_file.close()
