
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


app = Flask(__name__)
catboost_model = CatBoostClassifier()
xgboost_model = XGBClassifier()
lightgbm_model = LGBMClassifier()
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

app = Flask(__name__)
xgboost_model = XGBClassifier()
# Global variables
df = None
X_train, X_test, y_train, y_test = None, None, None, None

@app.route('/')
def index():
    global df
    top_rows = None
    if df is not None:
        top_rows = df.head().to_html(classes='table table-striped table-hover')
    return render_template('index1.html', df=df, top_rows=top_rows)



@app.route('/upload', methods=['POST'])
def upload():
    global df
    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            top_rows = df.head().to_html(classes='table table-striped table-hover')
            return render_template('index1.html', message='Dataset loaded successfully.', top_rows=top_rows)
        else:
            return render_template('index1.html', message='Please upload a valid CSV file.')
    else:
        return render_template('index1.html', message='File not uploaded.')


@app.route('/split', methods=['POST'])
def split():
    global df, X_train, X_test, y_train, y_test
    if df is not None:
        features = df.columns[:-1]
        X = df[features]
        y = df[df.columns[-1]]
        # Label encoding for binary classification
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

        message = f"Split completed successfully."
        
        df = None  # Reset df to prevent further processing
        X_train_shape = X_train.shape
        X_test_shape = X_test.shape
        y_train_shape = y_train.shape
        y_test_shape = y_test.shape

        return render_template('index1.html', message=message, 
                               X_train_shape=X_train_shape, X_test_shape=X_test_shape, 
                               y_train_shape=y_train_shape, y_test_shape=y_test_shape)
    else:
        return render_template('index1.html', message='Please upload and preprocess the dataset first.')


from sklearn.metrics import roc_auc_score

@app.route('/run_catboost', methods=['POST'])
def run_catboost():
    global X_train, X_test, y_train, y_test

    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        catboost_model = CatBoostClassifier()
        catboost_model.fit(X_train, y_train)
        y_pred = catboost_model.predict(X_test)
        y_probabilities = catboost_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        roc_auc = roc_auc_score(y_test, y_probabilities)*100

        return render_template('catboost_metrics.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)

    else:
        return render_template('index1.html', message='Please upload, preprocess, and split the dataset first.')



from xgboost import XGBClassifier

@app.route('/run_xgboost', methods=['POST'])
def run_xgboost():
    global X_train, X_test, y_train, y_test

    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        # Initialize XGBoost model
        xgboost_model = XGBClassifier()
        # Fit the model
        xgboost_model.fit(X_train, y_train)
        # Make predictions
        y_pred = xgboost_model.predict(X_test)
        y_probabilities = xgboost_model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        roc_auc = roc_auc_score(y_test, y_probabilities)*100

        return render_template('xgboost_metrics.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)

    else:
        return render_template('index1.html', message='Please upload, preprocess, and split the dataset first.')

@app.route('/run_lightgbm', methods=['POST'])
def run_lightgbm():
    global X_train, X_test, y_train, y_test

    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        lightgbm_model = LGBMClassifier()
        lightgbm_model.fit(X_train, y_train)
        y_pred = lightgbm_model.predict(X_test)
        y_probabilities = lightgbm_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)*100
        precision = precision_score(y_test, y_pred, average='weighted')*100
        recall = recall_score(y_test, y_pred, average='weighted')*100
        f1 = f1_score(y_test, y_pred, average='weighted')*100
        roc_auc = roc_auc_score(y_test, y_probabilities)*100

        return render_template('lightgbm_metrics.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)

    else:
        return render_template('index1.html', message='Please upload, preprocess, and split the dataset first.')

@app.route('/predict', methods=['POST'])
def predict():
    global lightgbm_model, X_train, y_train, df

    if lightgbm_model is not None:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                new_data = pd.read_csv(file)
                
                # Ensure that the model is trained before making predictions
                if X_train is not None and y_train is not None:
                    # Train the LightGBM model
                    lightgbm_model.fit(X_train, y_train)
                    
                    # Make predictions
                    predictions = lightgbm_model.predict(new_data)
                    
                    # Map predictions to meaningful labels
                    prediction_labels = ['Not Fraud' if pred == 1 else ' Fraud' for pred in predictions]
                    
                    # Append predicted values to the DataFrame
                    new_data['Predicted Values'] = prediction_labels

                    # Convert DataFrame to HTML table
                    predicted_table = new_data.to_html(classes='table table-striped table-hover', index=False)
                    
                    # Render the index1.html template with the updated DataFrame and predicted values
                    return render_template('index1.html', message='Prediction completed successfully.', top_rows=predicted_table)
                else:
                    return render_template('index1.html', message='Please train the LightGBM model first.')
            else:
                return render_template('index1.html', message='Please upload a valid CSV file for prediction.')
        else:
            return render_template('index1.html', message='File for prediction not uploaded.')
    else:
        return render_template('index1.html', message='Please train the LightGBM model first.')


if __name__ == '__main__':
    app.run(port=5101,debug=True)
