import pickle
import predict

model_file = 'pipeline.bin'

test_data = {
  "lead_source": "organic_search",
  "industry": "technology",
  "employment_status": "employed",
  "location": "europe",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0,
  "interaction_count": 7,
  "lead_score": 0.74
}

def model_call():
    with open(model_file, 'rb') as f:
        dv, model = pickle.load(f)

    X = dv.transform([test_data])
    y_pred = model.predict_proba(X)[0,1]

    print(y_pred)

if __name__ == "__main__":
    model_call()

