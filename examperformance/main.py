from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

df = pd.read_csv("student_academic.csv")
df.drop_duplicates(inplace=True)

x = df.drop(['GPA', 'Result', 'Grade'], axis=1)
y = df['GPA']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
clf = LinearRegression()
clf.fit(x_train, y_train)

x2 = df.drop(['GPA', 'Result', 'Grade'], axis=1)
y2 = df['Grade']
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=42)
clf2 = LogisticRegression(max_iter=1000)
clf2.fit(x2_train, y2_train)

x3 = df.drop(['GPA', 'Result', 'Grade'], axis=1)
y3 = df['Result']
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=42)
clf3 = LogisticRegression(max_iter=1000)
clf3.fit(x3_train, y3_train)

def generate_plot1():
    plt.figure(figsize=(8, 6))
    gpa_predict = clf.predict(x_test)
    plt.scatter(y_test, gpa_predict, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual GPA")
    plt.ylabel("Predicted GPA")
    plt.title("Actual vs Predicted GPA")
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_plot2():
    comparison2_df = pd.DataFrame({'Actual Grade': y2_test.values, 'Predicted Grade': clf2.predict(x2_test)})
    comparison2_df_melted = pd.melt(comparison2_df, var_name="Type", value_name="Grade")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=comparison2_df_melted, x="Grade", hue="Type", palette="Set2")
    plt.title("Actual vs Predicted Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.grid(True, axis='y')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_plot3():
    comparison3_df = pd.DataFrame({'Actual Result': y3_test.values, 'Predicted Result': clf3.predict(x3_test)})
    comparison3_df_melted = pd.melt(comparison3_df, var_name="Type", value_name="Result")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=comparison3_df_melted, x="Result", hue="Type", palette="Set1")
    plt.title("Actual vs Predicted Result Distribution")
    plt.xlabel("Result")
    plt.ylabel("Count")
    plt.grid(True, axis='y')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
        }
        input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
        .predictions {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f5e9;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }
        .prediction-item {
            margin: 8px 0;
        }
        img {
            max-width: 100%;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .plot-container {
            margin-top: 30px;
        }
        .plot-title {
            text-align: center;
            margin-bottom: 10px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Student Performance Prediction</h1>
        <form method="POST">
            <div class="form-group">
                <label for="math">Math Score</label>
                <input type="number" id="math" name="math" min="0" max="100" required>
            </div>
            <div class="form-group">
                <label for="science">Science Score</label>
                <input type="number" id="science" name="science" min="0" max="100" required>
            </div>
            <div class="form-group">
                <label for="computer">Computer Score</label>
                <input type="number" id="computer" name="computer" min="0" max="100" required>
            </div>
            <div class="form-group">
                <label for="studyhours">Study Hours</label>
                <input type="number" id="studyhours" name="studyhours" min="0" step="0.1" required>
            </div>
            <button type="submit">Predict Performance</button>
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        {% if predictions %}
            <div class="predictions">
                <h2>Prediction Results</h2>
                <div class="prediction-item"><strong>GPA:</strong> {{ predictions.gpa|round(2) }}</div>
                <div class="prediction-item"><strong>Grade:</strong> {{ predictions.grade }}</div>
                <div class="prediction-item"><strong>Result:</strong> {{ predictions.result }}</div>
            </div>
        {% endif %}

        <div class="plot-container">
            <div class="plot-title">GPA Prediction Accuracy</div>
            <img src="data:image/png;base64,{{ plot1 }}" alt="GPA Prediction">
            
            <div class="plot-title">Grade Prediction Distribution</div>
            <img src="data:image/png;base64,{{ plot2 }}" alt="Grade Prediction">
            
            <div class="plot-title">Result Prediction Distribution</div>
            <img src="data:image/png;base64,{{ plot3 }}" alt="Result Prediction">
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    predictions = None
    plot1 = generate_plot1()
    plot2 = generate_plot2()
    plot3 = generate_plot3()

    if request.method == 'POST':
        try:
            math = float(request.form['math'])
            science = float(request.form['science'])
            computer = float(request.form['computer'])
            studyhours = float(request.form['studyhours'])

            new_data = pd.DataFrame({
                'Math': [math],
                'Science': [science],
                'Computer': [computer],
                'StudyHours': [studyhours]
            })

            gpa_pred = clf.predict(new_data)[0]
            grade_pred = clf2.predict(new_data)[0]
            result_pred = clf3.predict(new_data)[0]

            predictions = {
                'gpa': gpa_pred,
                'grade': grade_pred,
                'result': result_pred
            }
        except ValueError as e:
            error = "Please enter valid numbers for all fields."
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template_string(html_template, 
                               error=error, 
                               predictions=predictions,
                               plot1=plot1,
                               plot2=plot2,
                               plot3=plot3)

if __name__ == '__main__':
    app.run(debug=True)