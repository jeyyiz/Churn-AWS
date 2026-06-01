import json
import os
import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "churn-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features: list[float]) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

st.title("Customer Churn Predictor")
st.write("Enter customer profile features to predict churn probability via SageMaker Endpoint.")

# Setup 10 User Inputs
age = st.slider("Age", 18, 100, 30)
gender = st.radio("Gender", ["Male", "Female"])
gender_encoded = 1.0 if gender == "Male" else 0.0

tenure = st.slider("Tenure (Months)", 0, 120, 24)
usage_frequency = st.slider("Usage Frequency", 0, 100, 10)
support_calls = st.slider("Support Calls", 0, 20, 2)
payment_delay = st.slider("Payment Delay (Days)", 0, 60, 0)
total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=150.0)
last_interaction = st.slider("Last Interaction (Days ago)", 0, 60, 5)

subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
sub_mapping = {"Basic": 0.0, "Standard": 1.0, "Premium": 2.0}
sub_encoded = sub_mapping[subscription_type]

contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
contract_mapping = {"Monthly": 0.0, "Quarterly": 1.0, "Annual": 2.0}
contract_encoded = contract_mapping[contract_length]

if st.button("Predict Churn", type="primary"):
    # Urutan HARUS sama dengan X_train
    features = [
        float(age),
        float(gender_encoded),
        float(tenure),
        float(usage_frequency),
        float(support_calls),
        float(payment_delay),
        float(total_spend),
        float(last_interaction),
        float(sub_encoded),
        float(contract_encoded)
    ]
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs = result["probabilities"][0]

        if label == "Churn":
            st.error(f"Prediction result: **{label}**")
        else:
            st.success(f"Prediction result: **{label}**")
            
        st.write("Class probabilities:")
        st.bar_chart({"probability": probs})
