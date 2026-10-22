from flask import Flask, request, render_template
import pandas as pd
import numpy as np

application = Flask(__name__)
app = application


# Home route
@app.route('/')
def home_page():
    return render_template('index.html')

# Prediction route
@app.route('/predictstudentdata', methods=['GET', 'POST'])
def predict_datapoint():
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline
    from src.utils import load_object

    if request.method == 'GET':
        return render_template('home.html')

    else:
        data = CustomData(
            Gender=int(request.form.get('Gender')),
            Accommodation_Type=int(request.form.get('Accommodation_Type')),
            Working_while_studying=int(request.form.get('Working_while_studying')),
            Participation_in_Social_Activities=int(request.form.get('Participation_in_Social_Activities')),
            Scholarship_Status=int(request.form.get('Scholarship_Status')),
            Academic_support=int(request.form.get('Academic_support')),

            Age_Range=int(request.form.get('Age_Range')),
            Level_of_Study=int(request.form.get('Level_of_Study')),
            Health_Challenges=int(request.form.get('Health_Challenges')),
            School_Activities_Stress=int(request.form.get('School_Activities_Stress')),
            Internet_Access=int(request.form.get('Internet_Access')),
            CGPA_Range=int(request.form.get('CGPA_Range')),
            Parental_Level_of_Education=int(request.form.get('Parental_Level_of_Education')),
            Hours_of_Study_per_Week=int(request.form.get('Hours_of_Study_per_Week')),
            Class_Attendance=int(request.form.get('Class_Attendance')),

            Faculty=request.form.get('Faculty'),
            Financial_Support_Source=request.form.get('Financial_Support_Source'),
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_dataframe()
        print("INPUT DATA:\n", pred_df)

        # Load pipeline with threshold
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Get the probability of dropout (Class 1)
        proba_dropout = predict_pipeline.predict_proba(pred_df)

        # If it's a list/array, extract the value. If it's a single number, keep it.
        if isinstance(proba_dropout, (list, np.ndarray)):
            proba_dropout = proba_dropout.flatten()[0]

        # Calculate confidence based on the prediction result
        if results[0] == 1:
            # If Predicted At-Risk, show the dropout probability
            confidence = round(proba_dropout * 100, 2)
        else:
            # If Predicted Stable, show the success probability (100 - dropout)
            confidence = round((1 - proba_dropout) * 100, 2)

        print("PREDICTION:", results)
        print("CONFIDENCE:", confidence)
        print("THRESHOLD USED:", predict_pipeline.get_threshold())


# Get top features influencing the prediction
        top_features = predict_pipeline.get_top_features(pred_df, top_n=5)
        feature_names = [
                feature[0]
                .replace('num_pipeline__', '')
                .replace('cat_pipeline__', '')
                .replace('_', ' ') 
                for feature in top_features
            ]
        if results[0] == 1:
            features_text = f"Key factors increasing dropout risk: {', '.join(feature_names)}" if feature_names else "Feature importance data unavailable."
        else:
            features_text = f"Key factors supporting academic success: {', '.join(feature_names)}" if feature_names else "Feature importance data unavailable."
        if results[0] == 1:
            prediction_text = "Status: At-Risk of Academic Attrition"
            recommendation = "Immediate academic counseling and financial review recommended."
            alert_class = "danger"
        else:
            prediction_text = "Status: Stable Academic Progress"
            recommendation = "Maintain current study habits and engagement levels."
            alert_class = "success" 

        # Final display string (including your confidence score and top features)
        display_message = f"{prediction_text} (Confidence: {confidence}%)<br>{features_text}"

        return render_template(
            'home.html',
            results=results[0],
            prediction_text=prediction_text,
            recommendation=recommendation,
            alert_class=alert_class,
            display_message=display_message
        )
        


if __name__ == "__main__":
    # app.run(host="127.0.0.1", port=5001, debug=True)
    app.run(host="0.0.0.0", port=5001, debug=True)