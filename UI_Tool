import streamlit as st
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Define company options
companies = ["FONZA", "SAMSUNG BIOLOGICS", "Boston Institute of Biotechnology"]

# Define research subjects and prompts
research_prompts = {
    "Manufacturing Capabilities": "Analyze the manufacturing capabilities, production capacity, and technical expertise of {company}. Focus on:\n1) Production facilities and capacity\n2) Manufacturing technologies\n3) Technical expertise and specializations\n4) Quality control systems",
    "Regulatory Compliance": "Evaluate the regulatory compliance status and quality standards of {company}. Focus on:\n1) Current certifications and approvals\n2) Compliance history\n3) Quality management systems\n4) Regulatory inspection outcomes",
    "Technology & Innovation": "Assess the technological capabilities and innovation profile of {company}. Focus on:\n1) Key technologies and equipment\n2) R&D capabilities\n3) Recent innovations and developments\n4) Technology partnerships",
    "Market Position": "Analyze the market position and competitive advantages of {company}. Focus on:\n1) Market presence and share\n2) Strategic partnerships\n3) Key differentiators\n4) Growth trajectory",
    "Services Portfolio": "Review the service portfolio and specializations of {company}. Focus on:\n1) Core service offerings\n2) Unique capabilities\n3) Service delivery model\n4) Client segments served"
}

# Streamlit UI
st.title("Company Research Tool")

# Company selection
company = st.selectbox("Company", companies, index=None, placeholder="Select a company")

# Research subject selection
st.subheader("Research Subject")
selected_subject = st.radio(
    label="Choose a research subject",
    options=list(research_prompts.keys()),
    index=None,
    key="research_subject",
    horizontal=False
)

# Auto-populated text area with company name dynamically replaced
if selected_subject and company:
    pre_defined_prompt = research_prompts[selected_subject].format(company=company)
else:
    pre_defined_prompt = ""

custom_query = st.text_area("Custom Query (Optional)", pre_defined_prompt, height=150)

# Submit button
if st.button("Analyze"):
    if not company or not selected_subject:
        st.error("Please select a company and a research subject.")
    else:
        final_query = f"{custom_query}"  # This includes the auto-populated or modified query
        st.info(f"Submitting query:\n\n{final_query}")

        # Backend API call (Replace with actual API URL and parameters)
        api_url = "https://your-backend-api.com/query"
        payload = {"query": final_query}

        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                st.success("Query successfully submitted!")
                st.write("Response:", response.json())
            else:
                st.error(f"Failed to submit query. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error submitting query: {e}")
