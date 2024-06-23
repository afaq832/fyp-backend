import joblib
from flask import Flask, session, redirect, url_for, request, flash, render_template,jsonify, send_from_directory
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import requests
import pandas as pd
from flask_cors import CORS
import random

from hugchat import hugchat
from hugchat.login import Login
app = Flask(__name__, static_url_path='', static_folder='static')

CORS(app, resources={r"/*": {"origins": ["*"]}})
DATABASE = 'app.db'
app.secret_key = 'your_secret_key'

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('Batsman_Data.csv')

def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

create_users_table()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print("hello")
        if 'username' not in session:
            flash('You need to be logged in to view this page.')
            print("Username not in session")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@login_required
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data['username']
    password = generate_password_hash(data['password'])

    try:
        conn = get_db_connection()
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Username already exists. Try a different one.'}), 409
    finally:
        if conn:
            conn.close()

    return jsonify({'message': 'Signup successful. Please login.'}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']

    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()

    if user is None:
        return jsonify({'error': 'Username not found.'}), 404
    elif not check_password_hash(user['password'], password):
        return jsonify({'error': 'Password is incorrect.'}), 401

    return jsonify({'message': 'Login successful.', 'username': username}), 200



@app.route('/chatbot', methods=['POST'])
def chat():
   data = request.get_json()
   userQuery = data['query']
   response = generate_response(userQuery,'aazhmeerchhapra@gmail.com','@@Zoo2002')
   return jsonify({
       'response':str(response)
   })






@login_required
@app.route('/logout')
def logout():
    print(session.pop('username', None))
    return redirect(url_for('index'))






# Load the model and model columns
rf_model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')



# Assuming your CSV files are in the same directory as your Flask app
BOWLER_DATA_PATH = 'Bowler_data.csv'
BATSMEN_DATA_PATH = 'Batsman_Data.csv'


def clean_opposition(dataframe):
    """Remove the 'v ' prefix from the Opposition column."""
    dataframe['Opposition'] = dataframe['Opposition'].str.replace('v ', '', regex=False)
    
    return dataframe


@app.route('/prediction', methods=['GET'])
def predict_frontend():

    bowler_df = pd.read_csv(BOWLER_DATA_PATH)
    batsmen_df = pd.read_csv(BATSMEN_DATA_PATH)
    
    bowler_df = clean_opposition(bowler_df)
    batsmen_df = clean_opposition(batsmen_df)

     # Extract player names and remove duplicates by converting to a set
    bowlers = set(bowler_df['Bowler'].unique())
    batsmen = set(batsmen_df['Batsman'].unique())
    
    # Combine both sets into one sorted list
    players = sorted(bowlers.union(batsmen))
    

    # Example of extracting unique team names after cleaning, for team selection dropdown
    teams = sorted(set(bowler_df['Opposition'].unique()).union(set(batsmen_df['Opposition'].unique())))

    # Pass the combined player names to the template
    return render_template('prediction.html', players=players, teams=teams)
    



def preprocess_user_input(data):
    # One-hot encode the 'Player' and 'Opposition' fields
    print(data)
    encoded_data = pd.get_dummies(data, columns=['Player', 'Opposition'])
    print("hekkoooo")
    # Create a DataFrame for missing columns with default value of 0
    missing_cols = {col: [0] * len(encoded_data) for col in model_columns if col not in encoded_data}
    missing_data = pd.DataFrame(missing_cols)
    
    # Concatenate the original encoded data with the missing columns DataFrame
    combined_data = pd.concat([encoded_data, missing_data], axis=1)
    
    # Ensure the order of columns matches the training data
    final_data = combined_data.reindex(columns=model_columns, fill_value=0)
    
    return final_data


@app.route('/predict', methods=['POST'])
def predict_runs():  
    data = request.get_json(force=True)
    try:
        input_data = pd.DataFrame([data])
        # print(input_data)
        
        preprocessed_input = preprocess_user_input(input_data)
        
        prediction = rf_model.predict(preprocessed_input)
        print(prediction)
        return jsonify({'predicted_runs': random.randint(0, 150), 'message': 'The Player {} is predicted to score {} runs against team {}'.format(data['Player'], prediction[0], data['Opposition'])})
    except Exception as e:
        return jsonify({'error': str(e)})

    


@app.route("/")
def index():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    return app.send_static_file('index.html')




if __name__ == '__main__':
    app.run(port=4000,debug=True)