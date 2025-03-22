import streamlit as st
import requests
import logging
import time
from datetime import datetime
import os
import streamlit as st
import subprocess
from langchain_core.messages import HumanMessage, SystemMessage

# Ensure required packages are installed in Streamlit environment
# Ensure the correct LangChain version is installed
#try:
#    from langchain_openai import ChatOpenAI  # Correct import for 2024+
#except ImportError:
#    st.warning("🔄 Installing missing dependencies...")
#    subprocess.run(["pip", "install", "--upgrade", "langchain-openai"], check=True)
#    from langchain_openai import ChatOpenAI  # Retry import after installation

#st.success("✅ ChatOpenAI successfully imported!")

# Logging setup
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

# Load the Button **Immediately**
analyze_button = st.button("Analyze")  # Button appears first



import requests
import time
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper

openai_api_key = st.secrets["OPENAI_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
google_cse_id = st.secrets["GOOGLE_CSE_ID"]
PPLX_API_KEY = st.secrets["PPLX_API_KEY"]

gpt4o_llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)
google_search = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=google_cse_id)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("query_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuerySystem")

# Default configuration for sources
DEFAULT_SOURCE_CONFIG = {
    "use_gpt4o": True,
    "use_perplexity": True,
    "use_google_search": True
}

def query_perplexity(query, step_info=""):
    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Querying Perplexity AI with: {query[:100]}...")
    start_time = time.time()

    url = "https://api.perplexity.ai/chat/completions"
    PERPLEXITY_API_KEY = PPLX_API_KEY
    #sonar-medium-online
    #"model": "sonar-reasoning-pro",
    payload = {
        #"model": "sonar-deep-research",
        "model": "sonar-reasoning",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    try:
        logger.info(f"{log_prefix}Sending request to Perplexity API...")
        response = requests.post(url, json=payload, headers=headers)
        logger.info(f"{log_prefix}Perplexity API response received. Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            elapsed_time = time.time() - start_time
            response_content = result["choices"][0]["message"]["content"]
            logger.info(f"{log_prefix}Perplexity AI responded successfully in {elapsed_time:.2f} seconds")
            logger.debug(f"{log_prefix}Perplexity response (first 100 chars): {response_content[:100]}...")
            return response_content
        else:
            logger.error(f"{log_prefix}Perplexity API error. Status Code: {response.status_code}")
            logger.error(f"{log_prefix}Full error response: {response.text}")
            return f"Error querying Perplexity: {response.status_code}"
    except Exception as e:
        logger.error(f"{log_prefix}Exception occurred while querying Perplexity: {str(e)}")
        return f"Exception: {str(e)}"


def query_all_sources(query, step_info="", source_config=None):
    # Use default config if none provided
    if source_config is None:
        source_config = DEFAULT_SOURCE_CONFIG

    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Starting multi-source query: {query[:100]}...")
    logger.info(f"{log_prefix}Source configuration: {source_config}")

    results = {}
    active_sources = []

    # Query GPT-4o if enabled
    if source_config.get("use_gpt4o", True):
        active_sources.append("GPT-4o")
        logger.info(f"{log_prefix}Querying GPT-4o...")
        start_time = time.time()
        try:
            results['GPT-4o'] = gpt4o_llm.invoke([HumanMessage(content=query)]).content
            elapsed_time = time.time() - start_time
            logger.info(f"{log_prefix}GPT-4o responded in {elapsed_time:.2f} seconds")
        except Exception as e:
            logger.error(f"{log_prefix}Error querying GPT-4o: {str(e)}")
            results['GPT-4o'] = f"Error: {str(e)}"
    else:
        logger.info(f"{log_prefix}Skipping GPT-4o (disabled in configuration)")
        results['GPT-4o'] = "Source disabled"

    # Query Perplexity if enabled
    if source_config.get("use_perplexity", True):
        active_sources.append("Perplexity")
        logger.info(f"{log_prefix}Querying Perplexity...")
        results['Perplexity'] = query_perplexity(query, step_info)
    else:
        logger.info(f"{log_prefix}Skipping Perplexity (disabled in configuration)")
        results['Perplexity'] = "Source disabled"

    # Query Google Custom Search if enabled
    if source_config.get("use_google_search", True):
        active_sources.append("Google Custom Search")
        logger.info(f"{log_prefix}Querying Google Custom Search...")
        start_time = time.time()
        try:
            results['Google Custom Search'] = google_search.run(query)
            elapsed_time = time.time() - start_time
            logger.info(f"{log_prefix}Google Custom Search responded in {elapsed_time:.2f} seconds")
        except Exception as e:
            logger.error(f"{log_prefix}Error querying Google Custom Search: {str(e)}")
            results['Google Custom Search'] = f"Error: {str(e)}"
    else:
        logger.info(f"{log_prefix}Skipping Google Custom Search (disabled in configuration)")
        results['Google Custom Search'] = "Source disabled"

    logger.info(f"{log_prefix}Completed querying all active sources: {', '.join(active_sources)}")
    return results


def should_break_down_query(query, step_info=""):
    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Evaluating if query should be broken down: {query[:100]}...")
    prompt = f"""
    Determine if the following query should be broken down into sub-queries.
    If yes, respond with 'Yes', otherwise respond with 'No'.
    Query: "{query}"
    """

    try:
        start_time = time.time()
        response = gpt4o_llm.invoke([HumanMessage(content=prompt)]).content.strip()
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix}Query breakdown decision: '{response}' (took {elapsed_time:.2f} seconds)")
        return response.lower() == 'yes'
    except Exception as e:
        logger.error(f"{log_prefix}Error determining if query should be broken down: {str(e)}")
        # Default to not breaking it down in case of error
        return False


def break_down_query(query, max_subqueries=10, step_info=""):
    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Breaking down query: {query[:100]}...")
    prompt = f"""
    Break down the following query into logically structured sub-queries.
    Provide AT MOST {max_subqueries} sub-queries.
    Focus on the most important aspects of the query.

    Query: "{query}"
    """

    try:
        start_time = time.time()
        sub_queries_raw = gpt4o_llm.invoke([HumanMessage(content=prompt)]).content
        elapsed_time = time.time() - start_time

        sub_queries = [sub_query.strip() for sub_query in sub_queries_raw.split('\n') if sub_query.strip()]

        # Enforce the maximum number of sub-queries
        if len(sub_queries) > max_subqueries:
            logger.info(f"{log_prefix}Limiting sub-queries from {len(sub_queries)} to {max_subqueries}")
            sub_queries = sub_queries[:max_subqueries]

        logger.info(f"{log_prefix}Query broken down into {len(sub_queries)} sub-queries (took {elapsed_time:.2f} seconds)")

        # Log each sub-query
        for i, sq in enumerate(sub_queries, 1):
            logger.info(f"{log_prefix}Sub-query {i}/{len(sub_queries)}: {sq[:100]}...")

        return sub_queries
    except Exception as e:
        logger.error(f"{log_prefix}Error breaking down query: {str(e)}")
        # Return the original query as a single item in case of error
        return [query]


def cross_validate_and_combine(query, results, step_info="", streamlit_callback=None):
    import time
    from langchain_core.messages import HumanMessage, SystemMessage

    log_prefix = f"{step_info} " if step_info else ""
    if streamlit_callback:
        streamlit_callback(f"{log_prefix}Starting content merge for query: {query[:100]}...")
    logger.info(f"{log_prefix}Starting content merge for query: {query[:100]}...")

    active_sources = [src for src, content in results.items() if content != "Source disabled"]
    logger.info(f"{log_prefix}Sources to merge: {', '.join(active_sources)}")
    if streamlit_callback:
        streamlit_callback(f"{log_prefix}Sources to merge: {', '.join(active_sources)}")

    for source in active_sources:
        result_content = results[source]
        char_count = len(result_content)
        word_count = len(result_content.split())
        has_error = result_content.startswith("Error:") or "Exception:" in result_content
        status = "ERROR" if has_error else "OK"

        logger.info(f"{log_prefix}Source: {source} | Status: {status} | {char_count} chars | ~{word_count} words")
        if streamlit_callback:
            streamlit_callback(f"{log_prefix}Source: {source} | Status: {status} | {char_count} chars | ~{word_count} words")

        if has_error:
            logger.error(f"{log_prefix}Source {source} returned error: {result_content[:200]}")
            if streamlit_callback:
                streamlit_callback(f"{log_prefix}Source {source} returned error.")
        else:
            logger.info(f"{log_prefix}Source {source} snippet: {result_content[:150]}...")

    effective_sources = [src for src in active_sources
                         if not results[src].startswith("Error") and
                            not "Exception:" in results[src]]

    logger.info(f"{log_prefix}Effective sources: {len(effective_sources)}/{len(active_sources)}")
    if streamlit_callback:
        streamlit_callback(f"{log_prefix}Effective sources: {len(effective_sources)}/{len(active_sources)}")

    if len(effective_sources) == 0:
        logger.warning(f"{log_prefix}No usable content, falling back to longest source (even if error).")
        if streamlit_callback:
            streamlit_callback(f"{log_prefix}No usable content. Using fallback.")
        best_source = max(active_sources, key=lambda src: len(results[src]))
        return results[best_source]

    if len(effective_sources) == 1:
        logger.info(f"{log_prefix}Only one usable source: {effective_sources[0]}. Returning as is.")
        if streamlit_callback:
            streamlit_callback(f"{log_prefix}Only one effective source. Skipping merge.")
        return results[effective_sources[0]]

    logger.info(f"{log_prefix}Merging {len(effective_sources)} sources")
    if streamlit_callback:
        streamlit_callback(f"{log_prefix}Merging content from {len(effective_sources)} sources...")

    combined_prompt = f"""
You are a research assistant. Merge the information from multiple sources into one comprehensive, structured report.

🔍 Requirements:
- Do NOT summarize, simplify, or remove technical or descriptive information.
- Preserve all unique phrasing, terminology, and contextual richness.
- If multiple sources mention the same info, include it once but retain extra context.
- Add proper citations (URL, article title, publisher) where relevant.
- Use inline citation style like [1], [2], etc.
- Include a References section at the end.

Main Query: "{query}"

Source Documents:
"""

    for source in effective_sources:
        combined_prompt += f"\n---\n{results[source]}\n"

    combined_prompt += """
Now write the full, detailed, citation-annotated report.
"""

    try:
        start_time = time.time()
        logger.info(f"{log_prefix}Sending merge prompt to GPT-4o...")
        if streamlit_callback:
            streamlit_callback(f"{log_prefix}Sending merge prompt to GPT-4o...")

        result = gpt4o_llm.invoke([
            SystemMessage(content="You are a citation-focused research assistant. Preserve all details and add source citations."),
            HumanMessage(content=combined_prompt)
        ]).content

        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix}Merge completed in {elapsed_time:.2f} seconds")
        if streamlit_callback:
            streamlit_callback(f"{log_prefix}Merge completed in {elapsed_time:.2f} seconds")

        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"{log_prefix}Error during merge: {error_msg}")
        if streamlit_callback:
            streamlit_callback(f"{log_prefix}Error during merge: {error_msg}")

        try:
            logger.warning(f"{log_prefix}Fallback: Using longest result from effective sources.")
            if streamlit_callback:
                streamlit_callback(f"{log_prefix}Fallback strategy: returning longest result.")
            best_source = max(effective_sources, key=lambda src: len(results[src]))
            return f"[Fallback to {best_source} only]\n\n{results[best_source]}"
        except:
            return f"Error during merge and fallback failed: {error_msg}"


############
def recursive_query(query, max_subqueries=10, source_config=None, streamlit_callback=None):
    if source_config is None:
        source_config = DEFAULT_SOURCE_CONFIG

    enabled_sources = [source for source, enabled in source_config.items() if enabled]

    if streamlit_callback:
        streamlit_callback("===== STARTING RECURSIVE QUERY =====")
        streamlit_callback(f"Main query: {query}")
        streamlit_callback(f"Maximum sub-queries: {max_subqueries}")
        streamlit_callback(f"Enabled sources: {', '.join(enabled_sources)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time_total = time.time()

    if streamlit_callback:
        streamlit_callback("\nSTEP 1: QUERY ANALYSIS")

    needs_breakdown = should_break_down_query(query, "[Step 1]")

    if streamlit_callback:
        streamlit_callback("\nSTEP 2: " + ("QUERY DECOMPOSITION" if needs_breakdown else "DIRECT QUERY PROCESSING"))

    if needs_breakdown:
        sub_queries = break_down_query(query, max_subqueries, "[Step 2]")

        if streamlit_callback:
            streamlit_callback(f"\nSTEP 3: PROCESSING {len(sub_queries)} SUB-QUERIES")

        combined_sub_results = {}

        for i, sub_query in enumerate(sub_queries, 1):
            if streamlit_callback:
                streamlit_callback(f"\n🔹 Sub-query {i}/{len(sub_queries)}: {sub_query}")

            results = query_all_sources(sub_query, f"[Step 3] Sub-query {i}", source_config)
            sub_result = cross_validate_and_combine(
                sub_query, results, f"[Step 3] Sub-query {i} Result"
            )
            combined_sub_results[sub_query] = sub_result

        if streamlit_callback:
            streamlit_callback("\nSTEP 4: FINAL INTEGRATION")

        combined_prompt = f"""
You are a research assistant generating a detailed formal report based on responses to a set of sub-questions.

🎯 Objective:
Generate a professional, detailed report that addresses the main query:
"{query}"

📌 Requirements:
- Create a formal report with clear, informative section headings.
- Do NOT summarize or shorten sub-query responses. Include all details.
- Rewrite each sub-query as a formal heading.
- Present the full response beneath each heading.
- Do not mention the word \"sub-query\" or use numbering like \"Sub-query 1\".
- At the end, include a \"References\" section if citations are mentioned.

The following are the sub-questions and their answers. Please incorporate each in full:
"""

        for sub_q, sub_result in combined_sub_results.items():
            combined_prompt += f"""
### {sub_q}
{sub_result}
"""

        combined_prompt += """
---
Now generate the final report based on the sections above.
Ensure the language is formal, objective, and professional.
"""

        try:
            final_result = gpt4o_llm.invoke([
                SystemMessage(content="You are a citation-focused research assistant."),
                HumanMessage(content=combined_prompt)
            ]).content
        except Exception as e:
            final_result = f"Error during final integration: {str(e)}"

    else:
        if streamlit_callback:
            streamlit_callback("\nSTEP 2: DIRECT QUERY PROCESSING")

        results = query_all_sources(query, "[Step 2]", source_config)

        if streamlit_callback:
            streamlit_callback("\nSTEP 3: COMBINING RESULTS")

        final_result = cross_validate_and_combine(query, results, "[Step 3]")

    if streamlit_callback:
        streamlit_callback("\nSTEP 4: FINALIZATION")
        streamlit_callback(f"Total execution time: {time.time() - start_time_total:.2f} seconds")

    try:
        result_filename = f"query_result_{timestamp}.txt"
        with open(result_filename, 'w') as f:
            f.write(f"Query: {query}\n\n")
            f.write(f"Enabled sources: {', '.join(enabled_sources)}\n\n")
            f.write(f"Result:\n{final_result}")
        if streamlit_callback:
            streamlit_callback(f"Result saved to {result_filename}")
    except Exception as e:
        if streamlit_callback:
            streamlit_callback(f"Error saving result to file: {str(e)}")

    if streamlit_callback:
        streamlit_callback("\nSTEP 5: COMPLETED")

    return final_result


if analyze_button:
    if not company or not selected_subject:
        st.error("Please select a company and a research subject.")
    else:
        final_query = custom_query
        st.info(f"📨 Submitting query:\n\n{final_query}")

        # Capture progress messages
        progress_log = []

        with st.status("Processing your query...", expanded=True) as status:
            def update_ui(msg):
                progress_log.append(msg)
                status.write(msg)

            update_ui("🛠️ Step 1: Validating and analyzing query...")
            time.sleep(1)

            update_ui("🔄 Step 2: Launching recursive query process...")
            try:
                final_result = recursive_query(
                    query=final_query,
                    max_subqueries=15,
                    source_config={
                        "use_gpt4o": True,
                        "use_perplexity": True,
                        "use_google_search": True
                    },
                    streamlit_callback=update_ui  # 👈 live progress updates
                )

                status.update(label="✅ Query Processed Successfully!", state="complete")
                st.success("Here is your full result:")
                st.write(final_result)

            except Exception as e:
                status.update(label="❌ Query Failed", state="error")
                st.error(f"An error occurred during processing: {e}")

