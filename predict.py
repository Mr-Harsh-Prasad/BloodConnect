import pandas as pd
import joblib


# -----------------------------------
# Load trained model
# -----------------------------------

model = joblib.load("model.pkl")


# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_screening(
    age,
    weight,
    hemoglobin,
    bp_systolic,
    bp_diastolic,
    heart_rate,
    temperature,
    previous_donation
):

    # Create input DataFrame

    data = pd.DataFrame([{
        "Age": age,
        "Weight": weight,
        "Hemoglobin": hemoglobin,
        "BP_Systolic": bp_systolic,
        "BP_Diastolic": bp_diastolic,
        "Heart_Rate": heart_rate,
        "Temperature_F": temperature,
        "Previous_Donation": previous_donation
    }])


    # Prediction

    prediction = model.predict(data)[0]


    # Probability

    probabilities = model.predict_proba(data)[0]

    classes = model.classes_

    probability_result = {}

    for class_name, probability in zip(classes, probabilities):

        probability_result[class_name] = round(
            float(probability) * 100,
            2
        )


    return {
        "result": prediction,
        "probabilities": probability_result
    }


# -----------------------------------
# Dictionary Function
# -----------------------------------

def predict_from_dict(data):

    return predict_screening(

        age=float(data["Age"]),

        weight=float(data["Weight"]),

        hemoglobin=float(data["Hemoglobin"]),

        bp_systolic=float(data["BP_Systolic"]),

        bp_diastolic=float(data["BP_Diastolic"]),

        heart_rate=float(data["Heart_Rate"]),

        temperature=float(data["Temperature_F"]),

        previous_donation=data["Previous_Donation"]

    )


# -----------------------------------
# Test Prediction
# -----------------------------------

if __name__ == "__main__":

    result = predict_screening(

        age=25,

        weight=65,

        hemoglobin=13.5,

        bp_systolic=120,

        bp_diastolic=80,

        heart_rate=72,

        temperature=98.4,

        previous_donation="No"

    )

    print("\nPrediction Result:")
    print(result)



    ''' python predict.py
    key to test the prediction function'''