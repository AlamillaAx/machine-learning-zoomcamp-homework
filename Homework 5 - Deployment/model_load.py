import pickle
import predict

model_file = 'pipeline.bin'

test_data = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

def model_call():
    with open(model_file, 'rb') as f:
        dv, model = pickle.load(f)

    X = dv.transform([test_data])
    y_pred = model.predict_proba(X)[0,1]

    print(y_pred)

if __name__ == "__main__":
    model_call()

