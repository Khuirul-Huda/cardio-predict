from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.get('/app')
def launch():
    return render_template('app.html')

@app.post('/app')
def predict():
    return render_template('app.html')

if __name__ == '__main__':
    app.run(debug=True)

def train_model():
    pass


