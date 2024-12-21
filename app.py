from flask import Flask, render_template, request, jsonify
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import joblib, os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.get('/app')
def launch():
    return render_template('app.html')

@app.post('/app')
def predict():
    user_input = {
        'age': int(request.form['age']),
        'sex': int(request.form['sex']),
        'cp': int(request.form['cp']),
        'trestbps': int(request.form['trestbps']),
        'chol': int(request.form['chol']),
        'fbs': int(request.form['fbs']),
        'restecg': int(request.form['restecg']),
        'thalach': int(request.form['thalach']),
        'exang': int(request.form['exang']),
        'oldpeak': float(request.form['oldpeak']),
        'slope': int(request.form['slope']),
        'ca': int(request.form['ca']),
        'thal': int(request.form['thal'])
    }

    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    user_input_scaled = preprocess_input(user_input, scaler, feature_columns)
    prediction = model.predict(user_input_scaled)
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    print(f"Prediction: {result}")
    return jsonify({'result': result})

# Step 1: Data Preparation
def load_data():
    url = "https://raw.githubusercontent.com/ajinkyalahade/Heart-Disease---Classifications-Machine-Learning-/refs/heads/master/heart.csv"
    return pd.read_csv(url)

# Step 2: Data Splitting
def split_data(df):
    X = df.drop(columns=["target"])  # Fitur
    y = df["target"]  # Target
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Data Scaling
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Step 4: Model Training
def train_model(X_train_scaled, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

# Step 5: Evaluation
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Step 6: Deployment - User Input Preprocessing
def preprocess_input(user_input, scaler, feature_columns):
    """
    Preprocess user input before passing to the model.
    """
    user_input = pd.DataFrame([user_input], columns=feature_columns)
    user_input_scaled = scaler.transform(user_input)
    return user_input_scaled

def create_and_dump_model():
     # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Train model
    model = train_model(X_train_scaled, y_train)

    # Evaluate model
    evaluate_model(model, X_test_scaled, y_test)

    # Save model and scaler for deployment
    joblib.dump(model, "random_forest_model.pkl")
    joblib.dump(scaler, "scaler.pkl")



if __name__ == '__main__':
    if (os.path.exists("random_forest_model.pkl") and os.path.exists("scaler.pkl")):
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
    else:
        create_and_dump_model()
        
    app.run(debug=True)


