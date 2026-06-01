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

# 1. Setup User Inputs berdasarkan skema fitur Churn
tenure = st.slider("Tenure (Months)", 0, 120, 24)
support_calls = st.slider("Support Calls", 0, 20, 2)
total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=150.0)

gender = st.radio("Gender", ["Male", "Female"])
gender_encoded = 1 if gender == "Male" else 0

subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
sub_mapping = {"Basic": 0.0, "Standard": 1.0, "Premium": 2.0}
sub_encoded = sub_mapping[subscription_type]

contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
contract_mapping = {"Monthly": 0.0, "Quarterly": 1.0, "Annual": 2.0}
contract_encoded = contract_mapping[contract_length]

if st.button("Predict Churn", type="primary"):
    # Gabungkan sesuai urutan FEATURE_NAMES di inference.py
    features = [
        float(tenure), 
        float(support_calls), 
        float(total_spend), 
        float(gender_encoded), 
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