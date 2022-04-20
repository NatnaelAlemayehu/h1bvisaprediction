from flask import Flask,render_template,Response, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mode
import re
from xgboost import XGBClassifier
import pickle

app=Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegistrationForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Register")

    def validate_usernmae(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError("That username already exists. Choose a different one.")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

class predictForm(FlaskForm):
    companychoices = [
        ('10','gth inc.'),
        ('15', 'umbel corp'),
        ('18', 'quicklogix, inc.'),
        ('20', 'westfield corporation'),
        ('23', 'mcchrystal group, llc'),
        ('24', 'quicklogix llc'),
        ('25', 'vricon, inc.'),
        ('26', 'burger king corporation'),
        ('30', 'goodman networks, inc.'),
        ('1', 'university of michigan'),
        ('8', 'cardiac science corporation'),
        ('12', 'mcchrystal group, llc'),
        ('13', 'sensorhound, inc.'),
        ('34', 'pronto general agency, ltd.'),
        ('35', 'natural american foods inc.'),
        ('36', 'parallels, inc.'),
        ('37', 'rancho la puerta llc')       
    ]

    occupationchoices = [
        ('1','computer occupations'),
        ('2','Mathematical Occupations'),
        ('3','Education Occupations'),
        ('4','Medical Occupations'),
        ('5','Management Occupation'),
        ('6','Marketing Occupation'),
        ('7','Financial Occupation'),
        ('8','Architecture & Engineering')       
    ]
    jobtype = SelectField(u'Full time job?', choices=[('1', 'Yes'), ('0', 'No')])
    companyname = SelectField(u'Company', choices=companychoices)   
    occupationcategory = SelectField(u'Category?', choices=occupationchoices)
    prevailingwage = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "prevailing wage"})
    
    submit = SubmitField("Make prediction")



@app.route('/', methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('predictpage'))
    return render_template('login.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('predictpage'))
    return render_template('login.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():  
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():   
    logout_user()
    return redirect(url_for('login')) 


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predictpage():
    form = predictForm()
    global companyname
    global wage
    global occupation
    global jobtype
    companyname = ""
    wage = ""
    occupation = ""
    jobtype = ""
    if form.validate_on_submit():
        pickled_model = pickle.load(open('XGB_Model_h1b.sav', 'rb'))
        companyarray = np.zeros(15)
        wagearray = np.zeros(15)
        occupationarray = np.zeros(15)
        jobtypearray = np.zeros(15)
        companyarray = np.insert(companyarray, form.companyname.data, form.companyname.data)
        wagearray = np.insert(wagearray, form.prevailingwage, form.prevailingwage)
        occupationarray = np.insert(occupationarray, form.occupationcategory.data, form.occupationcategory.data)
        jobtypearray = np.insert(jobtypearray, form.jobtype.data, form.jobtype.data)
        combinedarray = np.concatenate((wagearray, companyarray, occupationarray, jobtypearray), axis=0) 

        predictedresult = pickled_model.predict(combinedarray)
        
        if predictedresult == 0:
            result = 'Result: you are likely to be rejected!'  
        else:
            result = 'Result: you are likely to be granted H1B visa!'    

    return render_template('predict.html', form=form, modelresult=result)


if __name__=="__main__":
    app.run(debug=True)

