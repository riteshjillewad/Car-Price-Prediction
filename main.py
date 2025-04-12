import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')


# Home page route
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


# Index page route
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0], 2))


@app.route('/analysis', methods=['POST'])
@cross_origin()
def analysis():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Prediction
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    estimated_price = np.round(prediction[0], 2)

    # Determine condition based on kilometers driven
    if int(driven) < 20000:
        condition = "Excellent"
    elif int(driven) < 50000:
        condition = "Good"
    else:
        condition = "Average"

    # Pass the data to the analysis page
    return render_template('analysis.html', car_model=car_model, company=company, year=year, fuel_type=fuel_type,
                           driven=driven, condition=condition, estimated_price=estimated_price)


@app.route('/emi_calculator', methods=['GET', 'POST'])
def emi_calculator():
    if request.method == 'POST':
        car_price = float(request.form.get('car_price'))
        down_payment = float(request.form.get('downPayment'))
        loan_tenure = int(request.form.get('loanTenure'))
        interest_rate = float(request.form.get('interestRate'))

        # Calculate loan amount
        principal = car_price - down_payment

        # Calculate EMI
        monthly_interest_rate = interest_rate / 12 / 100
        number_of_months = loan_tenure * 12
        emi = (principal * monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_months) / \
              ((1 + monthly_interest_rate) ** number_of_months - 1)

        return render_template('emi.html', emi=np.round(emi, 2), car_price=car_price, down_payment=down_payment,
                               loan_tenure=loan_tenure, interest_rate=interest_rate, price=car_price)

    # Get the price from the query string for pre-filling the form
    estimated_price = request.args.get('price', '')
    return render_template('emi.html', price=estimated_price)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    car_models = sorted(car['name'].unique())
    return render_template('compare.html', car_models=car_models)


@app.route('/news', methods=['GET'])
def news():
    return render_template('news.html')


@app.route('/article1', methods=['GET'])
def article1():
    return render_template('article1.html')


@app.route('/article2', methods=['GET'])
def article2():
    return render_template('article2.html')


@app.route('/article3', methods=['GET'])
def article3():
    return render_template('article3.html')


@app.route('/article4', methods=['GET'])
def article4():
    return render_template('article4.html')


@app.route('/article5', methods=['GET'])
def article5():
    return render_template('article5.html')


@app.route('/comparison_result', methods=['POST'])
@cross_origin()
def compare_results():
    vehicle1 = request.form.get('vehicle1')
    vehicle2 = request.form.get('vehicle2')

    # Fetch details of both vehicles from the dataset
    details_vehicle1 = car[car['name'] == vehicle1].iloc[0]
    details_vehicle2 = car[car['name'] == vehicle2].iloc[0]

    # Example comparison logic: compare estimated prices
    if details_vehicle1['Price'] < details_vehicle2['Price']:
        recommendation = f"We recommend {vehicle1} as it is more cost-effective."
    else:
        recommendation = f"We recommend {vehicle2} as it is more cost-effective."

    return render_template('comparison_result.html', vehicle1=vehicle1, vehicle2=vehicle2,
                           details_vehicle1=details_vehicle1, details_vehicle2=details_vehicle2,
                           recommendation=recommendation)


if __name__ == '__main__':
    app.run(debug=True)
