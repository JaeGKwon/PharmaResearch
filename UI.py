import streamlit as st
import requests
import logging
import time
from datetime import datetime
import os
import streamlit as st
import subprocess

# Ensure required packages are installed in Streamlit environment
try:
    from langchain_openai import ChatOpenAI  # Attempt to import
except ImportError:
    st.warning("Installing missing dependencies...")
    subprocess.run(["pip", "install", "--upgrade", "langchain-openai"], check=True)
    from langchain_openai import ChatOpenAI  # Import again after installation
    
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

# Define function to process the query using your existing Python code
def process_query(query):
    # Replace this with your actual code logic
    st.info(f"Processing query:\n\n{query}")

    # Simulating API request (Replace with actual logic)
    time.sleep(2)  # Simulate a delay
    # response = f"Generated response for query:\n\n{query}" --> Replace the response with the following code
    max_subqueries = 10  # Set your limit here
    source_config = {
        "use_gpt4o": True,         # Set to False to disable GPT-4o
        "use_perplexity": True,    # Set to False to disable Perplexity
        "use_google_search": True  # Set to False to disable Google Custom Search
    }

    response = recursive_query(response, max_subqueries, source_config)

    return response

# Submit button
if st.button("Analyze"):
    if not company or not selected_subject:
        st.error("Please select a company and a research subject.")
    else:
        final_query = custom_query  # This includes the auto-populated or modified query
        st.info(f"Submitting query:\n\n{final_query}")

        # Call the function to process the query
        response = process_query(final_query)

        # Display the result
        st.success("Query processed successfully!")
        st.write(response)

import requests
import time
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI

openai_api_key = st.secrets["OPENAI_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
google_cse_id = st.secrets["GOOGLE_CSE_ID"]
pplx_api_key = st.secrets["PPLX_API_KEY"]

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

from langchain_core.messages import HumanMessage

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

def cross_validate_and_combine(query, results, step_info=""):
    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Starting cross-validation and reconciliation for query: {query[:100]}...")

    # Identify active sources (those not disabled)
    active_sources = [src for src, content in results.items() if content != "Source disabled"]
    logger.info(f"{log_prefix}Sources to reconcile: {', '.join(active_sources)}")

    # Log the length of results from each active source
    for source in active_sources:
        result_content = results[source]
        char_count = len(result_content)
        word_count = len(result_content.split())
        # Check if content starts with an error message
        has_error = result_content.startswith("Error:") or "Exception:" in result_content

        status = "ERROR" if has_error else "OK"
        logger.info(f"{log_prefix}Source: {source} | Status: {status} | {char_count} chars | ~{word_count} words")

        # Log a snippet of each source's response
        if has_error:
            logger.error(f"{log_prefix}Source {source} returned error: {result_content[:200]}")
        else:
            logger.info(f"{log_prefix}Source {source} snippet: {result_content[:150]}...")

    # Calculate effective sources (those without errors)
    effective_sources = [src for src in active_sources
                         if not results[src].startswith("Error") and
                            not "Exception:" in results[src]]

    logger.info(f"{log_prefix}Effective sources for reconciliation: {len(effective_sources)}/{len(active_sources)}")

    # If only one source is active or no effective sources, handle appropriately
    if len(effective_sources) == 0:
        logger.warning(f"{log_prefix}No effective sources available for reconciliation. Using best available source.")
        # Return the longest result even if it's an error
        best_source = max(active_sources, key=lambda src: len(results[src]))
        logger.info(f"{log_prefix}Using best available source: {best_source}")
        return results[best_source]
    elif len(effective_sources) == 1:
        logger.info(f"{log_prefix}Only one effective source ({effective_sources[0]}), skipping cross-validation")
        return results[effective_sources[0]]

    # Log reconciliation approach
    logger.info(f"{log_prefix}Starting detailed reconciliation process with {len(effective_sources)} sources")

    # Build prompt with only active and effective sources
    combined_prompt = f"""
    Cross-validate and combine results for the query: "{query}"

    Your task is to reconcile and synthesize information from multiple sources.
    Please analyze the following information sources carefully:
    """

    for source in effective_sources:
        combined_prompt += f"""
        {source}:
        {results[source]}
        """

    combined_prompt += f"""
    When creating your synthesis:
    1) Identify key points of agreement between sources
    2) Note any significant discrepancies or contradictions
    3) Highlight unique insights provided by each source
    4) Provide a coherent summary that integrates all reliable information
    5) Add a brief assessment of the relative reliability of each source when they conflict

    Provide a comprehensive, well-structured response that addresses the original query: "{query}"
    """

    try:
        start_time = time.time()
        logger.info(f"{log_prefix}Sending reconciliation prompt to GPT-4o (length: {len(combined_prompt)} chars)")

        # Log a sample of the consolidation prompt for debugging
        prompt_sample = combined_prompt[:500] + "..." if len(combined_prompt) > 500 else combined_prompt
        logger.debug(f"{log_prefix}Reconciliation prompt sample: {prompt_sample}")

        result = gpt4o_llm.invoke([HumanMessage(content=combined_prompt)]).content
        elapsed_time = time.time() - start_time

        # Log statistics about the reconciliation result
        result_char_count = len(result)
        result_word_count = len(result.split())
        result_para_count = len([p for p in result.split('\n\n') if p.strip()])

        logger.info(f"{log_prefix}Reconciliation completed in {elapsed_time:.2f} seconds")
        logger.info(f"{log_prefix}Reconciliation result statistics: {result_char_count} chars | ~{result_word_count} words | ~{result_para_count} paragraphs")
        logger.info(f"{log_prefix}Reconciliation result (first 200 chars): {result[:200]}...")

        # Log a signature indicating this was a reconciled result
        reconciliation_signature = f"[Reconciled from {len(effective_sources)} sources: {', '.join(effective_sources)}]"
        logger.info(f"{log_prefix}Reconciliation signature: {reconciliation_signature}")

        return result
    except Exception as e:
        error_msg = str(e)
        logger.error(f"{log_prefix}Error during reconciliation process: {error_msg}")

        # Fallback strategy - return the longest result from effective sources
        try:
            logger.warning(f"{log_prefix}Using fallback strategy: returning longest result from effective sources")
            best_source = max(effective_sources, key=lambda src: len(results[src]))
            logger.info(f"{log_prefix}Fallback to source: {best_source}")
            fallback_result = f"[Reconciliation failed, using {best_source} only] {results[best_source]}"
            return fallback_result
        except:
            return f"Error during cross-validation: {error_msg}. Failed to apply fallback strategy."

def recursive_query(query, max_subqueries=10, source_config=None):
    # Use default source configuration if none provided
    if source_config is None:
        source_config = DEFAULT_SOURCE_CONFIG

    # Log which sources are enabled
    enabled_sources = [source for source, enabled in source_config.items() if enabled]

    logger.info(f"===== STARTING RECURSIVE QUERY =====")
    logger.info(f"Main query: {query}")
    logger.info(f"Maximum sub-queries: {max_subqueries}")
    logger.info(f"Enabled sources: {', '.join(enabled_sources)}")

    # Create a timestamp for this query execution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Query execution ID: {timestamp}")

    start_time_total = time.time()

    # Define total steps for this process
    total_steps = 7  # Adjust based on your workflow

    # Clear step separation
    logger.info("\n" + "="*50)
    logger.info(f"STEP 1 OF {total_steps}: QUERY ANALYSIS")
    logger.info("="*50)

    # Step 1: Determine if query should be broken down
    step1_info = f"[Step 1/{total_steps}]"
    logger.info(f"{step1_info} Determining if query needs to be broken down")
    needs_breakdown = should_break_down_query(query, step1_info)

    # Clear step separation
    logger.info("\n" + "="*50)
    logger.info(f"STEP 2 OF {total_steps}: " + ("QUERY DECOMPOSITION" if needs_breakdown else "DIRECT QUERY PROCESSING"))
    logger.info("="*50)

    if needs_breakdown:
        # Step 2: Break down query
        step2_info = f"[Step 2/{total_steps}]"
        logger.info(f"{step2_info} Breaking down complex query")
        sub_queries = break_down_query(query, max_subqueries, step2_info)

        # Clear step separation
        logger.info("\n" + "="*50)
        logger.info(f"STEP 3 OF {total_steps}: SUB-QUERY PROCESSING")
        logger.info("="*50)

        # Step 3: Process sub-queries
        step3_info = f"[Step 3/{total_steps}]"
        logger.info(f"{step3_info} Beginning to process {len(sub_queries)} sub-queries")
        combined_sub_results = {}

        # Track metrics for each sub-query
        sub_query_metrics = {}

        for i, sub_query in enumerate(sub_queries, 1):
            sub_step_info = f"{step3_info} Sub-query {i}/{len(sub_queries)}"
            logger.info(f"\n{'-'*40}")
            logger.info(f"{sub_step_info}: {sub_query}")
            logger.info(f"{'-'*40}")
            sub_start_time = time.time()

            # Step 3a: Query sources for each sub-query
            results = query_all_sources(sub_query, f"{sub_step_info} - Source Querying", source_config)

            # Step 3b: Validate and combine results for each sub-query
            sub_result = cross_validate_and_combine(
                sub_query,
                results,
                f"{sub_step_info} - Result Validation"
            )

            combined_sub_results[sub_query] = sub_result

            # Calculate metrics for this sub-query
            sub_elapsed_time = time.time() - sub_start_time
            sub_query_metrics[sub_query] = {
                "execution_time": sub_elapsed_time,
                "result_length": len(sub_result),
                "word_count": len(sub_result.split()),
                "sources_used": [src for src, content in results.items()
                                if content != "Source disabled" and not content.startswith("Error")]
            }

            logger.info(f"{sub_step_info} completed in {sub_elapsed_time:.2f} seconds")
            logger.info(f"{sub_step_info} result length: {sub_query_metrics[sub_query]['result_length']} chars")
            logger.info(f"{sub_step_info} result word count: ~{sub_query_metrics[sub_query]['word_count']} words")

        # Clear step separation
        logger.info("\n" + "="*50)
        logger.info(f"STEP 4 OF {total_steps}: FINAL INTEGRATION")
        logger.info("="*50)

        # Step 4: Final combination of all sub-query results
        step4_info = f"[Step 4/{total_steps}]"
        logger.info(f"{step4_info} Combining all sub-query results (from {len(sub_queries)} sub-queries)")

        # Log statistics about sub-queries before integration
        total_sub_content = sum(len(res) for res in combined_sub_results.values())
        avg_sub_length = total_sub_content / len(combined_sub_results) if combined_sub_results else 0
        logger.info(f"{step4_info} Total content from all sub-queries: {total_sub_content} chars")
        logger.info(f"{step4_info} Average sub-query result length: {avg_sub_length:.1f} chars")

        # Prepare the prompt for final integration
        combined_prompt = f"""
        Combine and synthesize the following sub-query results to comprehensively answer the main query:

        MAIN QUERY: "{query}"

        You have been provided with results from {len(sub_queries)} sub-queries that together address
        different aspects of this main query. Your task is to synthesize these results into a cohesive
        and comprehensive response.

        SUB-QUERY RESULTS:
        """

        # Add each sub-query result with clear labeling
        for idx, (sq, res) in enumerate(combined_sub_results.items(), 1):
            combined_prompt += f"""

        SUB-QUERY {idx}: "{sq}"
        {'-' * 40}
        RESULT:
        {res}
        {'-' * 40}
        """

        combined_prompt += """

        YOUR SYNTHESIS SHOULD:
        1) Begin with a clear and concise summary of the main findings
        2) Organize information logically by topic rather than by sub-query
        3) Avoid unnecessary repetition while preserving important details
        4) Address any contradictions or inconsistencies between sub-query results
        5) Ensure all major aspects of the main query are addressed

        Provide a comprehensive, well-structured response that fully addresses the main query.
        """

        # Log information about the integration prompt
        logger.info(f"{step4_info} Integration prompt length: {len(combined_prompt)} chars")

        final_start_time = time.time()
        try:
            logger.info(f"{step4_info} Sending final integration request to GPT-4o")
            final_result = gpt4o_llm.invoke([HumanMessage(content=combined_prompt)]).content
            final_elapsed_time = time.time() - final_start_time

            # Log statistics about the final result
            final_char_count = len(final_result)
            final_word_count = len(final_result.split())
            final_para_count = len([p for p in final_result.split('\n\n') if p.strip()])
            compression_ratio = final_char_count / total_sub_content if total_sub_content > 0 else 0

            logger.info(f"{step4_info} Final integration completed in {final_elapsed_time:.2f} seconds")
            logger.info(f"{step4_info} Final result statistics:")
            logger.info(f"{step4_info} - Character count: {final_char_count}")
            logger.info(f"{step4_info} - Approximate word count: {final_word_count}")
            logger.info(f"{step4_info} - Paragraph count: {final_para_count}")
            logger.info(f"{step4_info} - Compression ratio: {compression_ratio:.2f} (final size / total inputs)")
            logger.info(f"{step4_info} - Result snippet: {final_result[:200]}...")
        except Exception as e:
            logger.error(f"{step4_info} Error in final integration: {str(e)}")

            # Attempt fallback strategy
            logger.warning(f"{step4_info} Attempting fallback integration strategy")
            try:
                # Simple concatenation fallback
                fallback_result = f"MAIN QUERY: {query}\n\n"
                for idx, (sq, res) in enumerate(combined_sub_results.items(), 1):
                    fallback_result += f"SUB-QUERY {idx}: {sq}\n{'-' * 40}\n{res}\n\n"

                fallback_result += "\n[Note: This is a simple concatenation of sub-query results due to integration error]"
                final_result = fallback_result
                logger.info(f"{step4_info} Fallback integration applied successfully")
            except Exception as fallback_error:
                logger.error(f"{step4_info} Fallback integration also failed: {str(fallback_error)}")
                final_result = f"Error during final integration: {str(e)}. Fallback also failed."
    else:
        # Alternative flow when no breakdown is needed
        # Step 2 (in this path): Direct query of all sources
        step2_direct_info = f"[Step 2/{total_steps}]"
        logger.info(f"{step2_direct_info} Processing query directly (no sub-queries)")
        results = query_all_sources(query, step2_direct_info, source_config)

        # Clear step separation
        logger.info("\n" + "="*50)
        logger.info(f"STEP 3 OF {total_steps}: RESULT INTEGRATION")
        logger.info("="*50)

        # Step 3 (in this path): Cross-validate and combine
        step3_direct_info = f"[Step 3/{total_steps}]"
        final_result = cross_validate_and_combine(query, results, step3_direct_info)

        # Since we're skipping step 4 in this path, add a placeholder
        logger.info("\n" + "="*50)
        logger.info(f"STEP 4 OF {total_steps}: SKIPPED (NOT NEEDED FOR DIRECT QUERIES)")
        logger.info("="*50)

    # Clear step separation
    logger.info("\n" + "="*50)
    logger.info(f"STEP 5 OF {total_steps}: FINALIZATION")
    logger.info("="*50)

    # Step 5: Calculate stats and wrap up
    step5_info = f"[Step 5/{total_steps}]"
    total_elapsed_time = time.time() - start_time_total
    logger.info(f"{step5_info} Total execution time: {total_elapsed_time:.2f} seconds")

    # Clear step separation
    logger.info("\n" + "="*50)
    logger.info(f"STEP 6 OF {total_steps}: RESULT STORAGE")
    logger.info("="*50)

    # Step 6: Save results
    step6_info = f"[Step 6/{total_steps}]"
    result_filename = f"query_result_{timestamp}.txt"
    try:
        with open(result_filename, 'w') as f:
            f.write(f"Query: {query}\n\n")
            f.write(f"Enabled sources: {', '.join(enabled_sources)}\n\n")
            f.write(f"Result:\n{final_result}")
        logger.info(f"{step6_info} Result saved to {result_filename}")
    except Exception as e:
        logger.error(f"{step6_info} Error saving result to file: {str(e)}")

    # Clear step separation
    logger.info("\n" + "="*50)
    logger.info(f"STEP 7 OF {total_steps}: COMPLETION")
    logger.info("="*50)

    # Step 7: Return results
    step7_info = f"[Step 7/{total_steps}]"
    logger.info(f"{step7_info} Query execution completed successfully")
    logger.info(f"\n===== RECURSIVE QUERY COMPLETED in {total_elapsed_time:.2f} seconds =====")

    return final_result

# Example usage
if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("="*30 + " STARTING NEW QUERY EXECUTION " + "="*30)
    logger.info("="*80 + "\n")

    main_query = "Provide a comprehensive evaluation of recent FDA inspection history, regulatory compliance, and sustainability practices for Thermo Fisher Scientific."
    max_subqueries = 10  # Set your limit here

    # Configure which sources to use (all enabled by default)
    source_config = {
        "use_gpt4o": True,         # Set to False to disable GPT-4o
        "use_perplexity": True,    # Set to False to disable Perplexity
        "use_google_search": True  # Set to False to disable Google Custom Search
    }

    try:
        # To use all sources (default)
        # final_summary = recursive_query(main_query, max_subqueries)

        # To use selected sources
        final_summary = recursive_query(main_query, max_subqueries, source_config)

        # Example: Use only GPT-4o
        # final_summary = recursive_query(main_query, max_subqueries, {
        #     "use_gpt4o": True,
        #     "use_perplexity": False,
        #     "use_google_search": False
        # })

        print("\n\nFINAL RESULT:")
        print(final_summary)
    except Exception as e:
        logger.error(f"Fatal error in query execution: {str(e)}")
        print(f"Error: {str(e)}")
