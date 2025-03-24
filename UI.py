import streamlit as st
import requests
import logging
import time
import os
import subprocess
import concurrent.futures
import asyncio
import nest_asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime

# Enable nested asyncio to allow asyncio in Streamlit environment
nest_asyncio.apply()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Define company options
companies = ["LONZA", "SAMSUNG BIOLOGICS", "WuXi Biologics", "Piramal Pharma Solutions", "Boston Institute of Biotechnology"]

# Define research subjects and prompts
research_prompts = {
    "Manufacturing Capabilities": "Analyze the manufacturing capabilities, production capacity, and technical expertise of {company}. Focus on:\n1) Production facilities and capacity\n2) Manufacturing technologies\n3) Technical expertise and specializations\n4) Quality control systems",
    "Regulatory Compliance": "Evaluate the regulatory compliance status and quality standards of {company}. Focus on:\n1) Current certifications and approvals\n2) Compliance history\n3) Quality management systems\n4) Regulatory inspection outcomes",
    "Technology & Innovation": "Assess the technological capabilities and innovation profile of {company}. Focus on:\n1) Key technologies and equipment\n2) R&D capabilities\n3) Recent innovations and developments\n4) Technology partnerships",
    "Market Position": "Analyze the market position and competitive advantages of {company}. Focus on:\n1) Market presence and share\n2) Strategic partnerships\n3) Key differentiators\n4) Growth trajectory",
    "Services Portfolio": "Review the service portfolio and specializations of {company}. Focus on:\n1) Core service offerings\n2) Unique capabilities\n3) Service delivery model\n4) Client segments served"
}

# Title (same size as subtitle)
st.markdown("## Company Research Tool")

# Company selection
company = st.selectbox("Company", companies, index=None, placeholder="Select a company")

# Research subject selection
st.markdown("## Research Subject")
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

custom_query = st.text_area("Custom Query (Optional). You can modify the query", pre_defined_prompt, height=150)

# Concurrency settings
max_concurrent_subqueries = st.sidebar.slider("Max Concurrent Sub-queries", min_value=1, max_value=20, value=5, 
                                             help="Maximum number of sub-queries to process in parallel. Higher values can improve speed but might hit API rate limits.")

max_concurrent_sources = st.sidebar.slider("Max Concurrent Source Queries", min_value=1, max_value=5, value=3,
                                         help="Maximum number of sources to query in parallel for each sub-query.")

# Load the Button **Immediately**
analyze_button = st.button("Analyze")

# API keys and setup
openai_api_key = st.secrets["OPENAI_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
google_cse_id = st.secrets["GOOGLE_CSE_ID"]
PPLX_API_KEY = st.secrets["PPLX_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper

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

# Semaphore to control number of concurrent HTTP requests to avoid rate limiting
http_semaphore = asyncio.Semaphore(10)

async def query_perplexity_async(query, step_info=""):
    """Asynchronous version of Perplexity query function"""
    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Querying Perplexity AI with: {query[:100]}...")
    start_time = time.time()

    url = "https://api.perplexity.ai/chat/completions"
    PERPLEXITY_API_KEY = PPLX_API_KEY
    
    payload = {
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
        
        # Use semaphore to limit concurrent requests
        async with http_semaphore:
            # Use asyncio-compatible HTTP client for better performance
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    status_code = response.status_code if hasattr(response, 'status_code') else response.status
                    
                    logger.info(f"{log_prefix}Perplexity API response received. Status Code: {status_code}")

                    if status_code == 200:
                        import json
                        result = json.loads(response_text)
                        elapsed_time = time.time() - start_time
                        response_content = result["choices"][0]["message"]["content"]
                        logger.info(f"{log_prefix}Perplexity AI responded successfully in {elapsed_time:.2f} seconds")
                        logger.debug(f"{log_prefix}Perplexity response (first 100 chars): {response_content[:100]}...")
                        return response_content
                    else:
                        logger.error(f"{log_prefix}Perplexity API error. Status Code: {status_code}")
                        logger.error(f"{log_prefix}Full error response: {response_text}")
                        return f"Error querying Perplexity: {status_code}"
    except Exception as e:
        logger.error(f"{log_prefix}Exception occurred while querying Perplexity: {str(e)}")
        return f"Exception: {str(e)}"

# Fallback to synchronous version if needed
def query_perplexity(query, step_info=""):
    """Synchronous version for fallback"""
    return asyncio.run(query_perplexity_async(query, step_info))

async def query_gpt4o_async(query, step_info=""):
    """Asynchronous wrapper for GPT-4o query"""
    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Querying GPT-4o...")
    start_time = time.time()
    
    try:
        # Use a thread pool for the blocking LangChain call
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: gpt4o_llm.invoke([HumanMessage(content=query)]).content
            )
            
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix}GPT-4o responded in {elapsed_time:.2f} seconds")
        return response
    except Exception as e:
        logger.error(f"{log_prefix}Error querying GPT-4o: {str(e)}")
        return f"Error: {str(e)}"

async def query_google_search_async(query, step_info=""):
    """Asynchronous wrapper for Google search"""
    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Querying Google Custom Search...")
    start_time = time.time()
    
    try:
        # Use a thread pool for the blocking Google search call
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: google_search.run(query)
            )
            
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix}Google Custom Search responded in {elapsed_time:.2f} seconds")
        return response
    except Exception as e:
        logger.error(f"{log_prefix}Error querying Google Custom Search: {str(e)}")
        return f"Error: {str(e)}"

async def query_all_sources_async(query, step_info="", source_config=None):
    """Asynchronous version that queries all sources in parallel"""
    # Use default config if none provided
    if source_config is None:
        source_config = DEFAULT_SOURCE_CONFIG

    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Starting parallel multi-source query: {query[:100]}...")
    logger.info(f"{log_prefix}Source configuration: {source_config}")

    results = {}
    tasks = []

    # Create tasks for all enabled sources
    if source_config.get("use_gpt4o", True):
        tasks.append(query_gpt4o_async(query, f"{step_info} GPT-4o"))
        results['GPT-4o'] = None  # Placeholder to maintain order
    else:
        results['GPT-4o'] = "Source disabled"

    if source_config.get("use_perplexity", True):
        tasks.append(query_perplexity_async(query, f"{step_info} Perplexity"))
        results['Perplexity'] = None  # Placeholder
    else:
        results['Perplexity'] = "Source disabled"

    if source_config.get("use_google_search", True):
        tasks.append(query_google_search_async(query, f"{step_info} Google"))
        results['Google Custom Search'] = None  # Placeholder
    else:
        results['Google Custom Search'] = "Source disabled"

    # Wait for all tasks to complete
    if tasks:
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assign results to the corresponding sources
        i = 0
        for source in results:
            if results[source] is None:  # This was a placeholder for an active source
                if isinstance(completed_tasks[i], Exception):
                    logger.error(f"{log_prefix}Error in {source}: {str(completed_tasks[i])}")
                    results[source] = f"Error: {str(completed_tasks[i])}"
                else:
                    results[source] = completed_tasks[i]
                i += 1

    active_sources = [src for src, content in results.items() if content != "Source disabled"]
    logger.info(f"{log_prefix}Completed querying all active sources in parallel: {', '.join(active_sources)}")
    return results

# Synchronous wrapper for query_all_sources_async
def query_all_sources_parallel(query, step_info="", source_config=None):
    """Synchronous wrapper that runs the async function"""
    return asyncio.run(query_all_sources_async(query, step_info, source_config))

async def should_break_down_query_async(query, step_info=""):
    """Asynchronous version of query breakdown decision"""
    log_prefix = f"{step_info} " if step_info else ""
    logger.info(f"{log_prefix}Evaluating if query should be broken down: {query[:100]}...")
    prompt = f"""
    Determine if the following query should be broken down into sub-queries.
    If yes, respond with 'Yes', otherwise respond with 'No'.
    Query: "{query}"
    """

    try:
        start_time = time.time()
        
        # Use a thread pool for the blocking LangChain call
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: gpt4o_llm.invoke([HumanMessage(content=prompt)]).content.strip()
            )
            
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix}Query breakdown decision: '{response}' (took {elapsed_time:.2f} seconds)")
        return response.lower() == 'yes'
    except Exception as e:
        logger.error(f"{log_prefix}Error determining if query should be broken down: {str(e)}")
        # Default to not breaking it down in case of error
        return False

# Synchronous wrapper
def should_break_down_query(query, step_info=""):
    """Synchronous wrapper for the async function"""
    return asyncio.run(should_break_down_query_async(query, step_info))

async def break_down_query_async(query, max_subqueries=10, step_info=""):
    """Asynchronous version of query breakdown"""
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
        
        # Use a thread pool for the blocking LangChain call
        with concurrent.futures.ThreadPoolExecutor() as executor:
            sub_queries_raw = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: gpt4o_llm.invoke([HumanMessage(content=prompt)]).content
            )
            
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

# Synchronous wrapper
def break_down_query(query, max_subqueries=10, step_info=""):
    """Synchronous wrapper for the async function"""
    return asyncio.run(break_down_query_async(query, max_subqueries, step_info))

async def cross_validate_and_combine_async(query, results, step_info="", streamlit_callback=None):
    """Asynchronous version of cross validate and combine"""
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
- IMPORTANT: INCLUDE ALL INFORMATION in full detail. DO NOT summarize or condense content.
- Retain 100% of the technical details, numerical data, and specialized terminology.
- Preserve all unique phrasing, examples, explanations, and contextual elements.
- Create a structure that accommodates all information without omission.
- If multiple sources contain the same information, include all versions with their nuances.
- Add proper citations (URL, article title, publisher) where relevant.
- Use inline citation style like [1], [2], etc.
- Include a References section at the end.
- The final document should contain ALL the information from ALL sources, fully presented.

Main Query: "{query}"

Source Documents:
"""

    for source in effective_sources:
        combined_prompt += f"\n---\n{results[source]}\n"

    combined_prompt += """
Now write the full, detailed, citation-annotated report.
Include ALL information from ALL sources. 
DO NOT summarize or condense any content.
Comprehensiveness is the top priority.
"""

    try:
        start_time = time.time()
        logger.info(f"{log_prefix}Sending merge prompt to GPT-4o...")
        if streamlit_callback:
            streamlit_callback(f"{log_prefix}Sending merge prompt to GPT-4o...")

        # Use a thread pool for the blocking LangChain call
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: gpt4o_llm.invoke([
                    SystemMessage(content="You are a citation-focused research assistant. Your primary goal is to preserve ALL details and information from ALL sources without summarization. Comprehensiveness is more important than conciseness."),
                    HumanMessage(content=combined_prompt)
                ]).content
            )

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

# Synchronous wrapper
def cross_validate_and_combine(query, results, step_info="", streamlit_callback=None):
    """Synchronous wrapper for the async function"""
    return asyncio.run(cross_validate_and_combine_async(query, results, step_info, streamlit_callback))

async def process_subquery(sub_query, idx, total, source_config, step_info, streamlit_callback):
    """Process a single subquery asynchronously"""
    sub_step_info = f"{step_info} Sub-query {idx}/{total}"
    
    if streamlit_callback:
        streamlit_callback(f"\n🔹 Processing Sub-query {idx}/{total}: {sub_query}")
    
    # Query all sources in parallel
    results = await query_all_sources_async(sub_query, sub_step_info, source_config)
    
    # Combine results
    sub_result = await cross_validate_and_combine_async(
        sub_query, 
        results, 
        f"{sub_step_info} Result",
        streamlit_callback
    )
    
    if streamlit_callback:
        streamlit_callback(f"✅ Completed Sub-query {idx}/{total}")
    
    return sub_query, sub_result

async def recursive_query_async(query, max_subqueries=10, source_config=None, streamlit_callback=None, 
                              max_concurrent_subqueries=5, max_concurrent_sources=3):
    """Fully asynchronous recursive query implementation"""
    if source_config is None:
        source_config = DEFAULT_SOURCE_CONFIG

    enabled_sources = [source for source, enabled in source_config.items() if enabled]

    if streamlit_callback:
        streamlit_callback("===== STARTING PARALLEL RECURSIVE QUERY =====")
        streamlit_callback(f"Main query: {query}")
        streamlit_callback(f"Maximum sub-queries: {max_subqueries}")
        streamlit_callback(f"Enabled sources: {', '.join(enabled_sources)}")
        streamlit_callback(f"Max concurrent sub-queries: {max_concurrent_subqueries}")
        streamlit_callback(f"Max concurrent source queries: {max_concurrent_sources}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time_total = time.time()

    if streamlit_callback:
        streamlit_callback("\nSTEP 1: QUERY ANALYSIS")

    needs_breakdown = await should_break_down_query_async(query, "[Step 1]")

    if streamlit_callback:
        streamlit_callback("\nSTEP 2: " + ("QUERY DECOMPOSITION" if needs_breakdown else "DIRECT QUERY PROCESSING"))

    if needs_breakdown:
        sub_queries = await break_down_query_async(query, max_subqueries, "[Step 2]")

        if streamlit_callback:
            streamlit_callback(f"\nSTEP 3: PROCESSING {len(sub_queries)} SUB-QUERIES IN PARALLEL")

        # Create a semaphore to limit concurrent subquery processing
        subquery_semaphore = asyncio.Semaphore(max_concurrent_subqueries)
        
        # Process subqueries with concurrency control
        async def process_with_semaphore(sub_query, idx, total):
            async with subquery_semaphore:
                return await process_subquery(
                    sub_query, 
                    idx, 
                    total, 
                    source_config, 
                    "[Step 3]", 
                    streamlit_callback
                )
        
        # Create tasks for all subqueries
        tasks = [process_with_semaphore(sq, i+1, len(sub_queries)) 
                for i, sq in enumerate(sub_queries)]
        
        # Execute all tasks and gather results
        sub_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        combined_sub_results = {}
        for i, result in enumerate(sub_results):
            if isinstance(result, Exception):
                logger.error(f"[Step 3] Error processing sub-query {i+1}: {str(result)}")
                if streamlit_callback:
                    streamlit_callback(f"❌ Error in sub-query {i+1}: {str(result)}")
                # Use the original sub-query as key with error message as value
                combined_sub_results[sub_queries[i]] = f"Error processing this sub-query: {str(result)}"
            else:
                sub_query, sub_result = result
                combined_sub_results[sub_query] = sub_result

        if streamlit_callback:
            streamlit_callback("\nSTEP 4: FINAL INTEGRATION")

        combined_prompt = f"""
You are a research assistant generating a detailed formal report based on responses to a set of sub-questions.

🎯 Objective:
Generate a comprehensive professional report that addresses the main query:
"{query}"

📌 Requirements:
- Create a formal report with clear, informative section headings.
- CRITICAL: INCLUDE ALL INFORMATION FROM ALL SUB-QUERY RESPONSES WITH NO OMISSIONS.
- Do NOT summarize, condense, or shorten any content from sub-query responses.
- Include every detail, example, explanation, and nuance from the original responses.
- Rewrite each sub-query as a formal heading.
- Present the complete, unabridged response beneath each heading.
- Keep all technical details, specialized terminology, numerical data, and examples intact.
- Do not mention the word "sub-query" or use numbering like "Sub-query 1".
- At the end, include a "References" section if citations are mentioned.
- Prioritize comprehensive inclusion of all information over conciseness.

The following are the sub-questions and their answers. Please incorporate each in full without any reduction in content or detail:
"""

        for sub_q, sub_result in combined_sub_results.items():
            combined_prompt += f"""
### {sub_q}
{sub_result}
"""

        combined_prompt += """
---
Now generate the final report based on the sections above.
IMPORTANT: Include ALL information from ALL sections without summarization or condensing.
Ensure the language is formal, objective, and professional.
Do not omit any details, examples, or explanations from the original responses.
"""

        try:
            if streamlit_callback:
                streamlit_callback("Generating final integrated report...")
                
            # Use a thread pool for the blocking LangChain call
            with concurrent.futures.ThreadPoolExecutor() as executor:
                final_result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: gpt4o_llm.invoke([
                        SystemMessage(content="You are a citation-focused research assistant. Your primary objective is to include ALL information from ALL sources without summarization or condensing. Comprehensiveness is more important than conciseness."),
                        HumanMessage(content=combined_prompt)
                    ]).content
                )
        except Exception as e:
            final_result = f"Error during final integration: {str(e)}"
            logger.error(f"[Step 4] {final_result}")
            if streamlit_callback:
                streamlit_callback(f"❌ {final_result}")

    else:
        if streamlit_callback:
            streamlit_callback("\nSTEP 2: DIRECT PARALLEL QUERY PROCESSING")

        # Query all sources in parallel
        results = await query_all_sources_async(query, "[Step 2]", source_config)

        if streamlit_callback:
            streamlit_callback("\nSTEP 3: COMBINING RESULTS")

        # Combine results
        final_result = await cross_validate_and_combine_async(query, results, "[Step 3]", streamlit_callback)

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

# Synchronous wrapper for the main entry point
def recursive_query(query, max_subqueries=10, source_config=None, streamlit_callback=None, 
                   max_concurrent_subqueries=5, max_concurrent_sources=3):
    """Synchronous wrapper for the main async function"""
    return asyncio.run(recursive_query_async(
        query, 
        max_subqueries, 
        source_config, 
        streamlit_callback, 
        max_concurrent_subqueries, 
        max_concurrent_sources
    ))



#############



if analyze_button:
    if not company or not selected_subject:
        st.error("Please select a company and a research subject.")
    else:
        final_query = custom_query
        st.info(f"📨 Submitting query:\n\n{final_query}")

        # Capture progress messages
        progress_log = []
        execution_completed = False
        execution_result = None
        execution_error = None

        with st.status("Processing your query...", expanded=True) as status:
            def update_ui(msg):
                progress_log.append(msg)
                status.write(msg)

            update_ui("🛠️ Step 1: Validating and analyzing query...")
            time.sleep(1)

            update_ui("🔄 Step 2: Launching parallel recursive query process...")
            try:
                final_result = recursive_query(
                    query=final_query,
                    max_subqueries=20,
                    source_config={
                        "use_gpt4o": True,
                        "use_perplexity": True,
                        "use_google_search": True
                    },
                    streamlit_callback=update_ui,  # 👈 live progress updates
                    max_concurrent_subqueries=max_concurrent_subqueries,  # Use the slider value
                    max_concurrent_sources=max_concurrent_sources  # Use the slider value
                )
                
                execution_completed = True
                execution_result = final_result
                
                status.update(label="✅ Query Processed Successfully!", state="complete")

            except Exception as e:
                import traceback
                execution_error = traceback.format_exc()
                status.update(label="❌ Query Failed", state="error")
                update_ui(f"Error: {str(e)}")

        # Display results AFTER the status container is closed
        if execution_completed:
            st.success("Here is your full result:")
            st.write(execution_result)

            # Add download button for the result
            result_txt = f"Query: {final_query}\n\n{execution_result}"
            st.download_button(
                label="Download Results",
                data=result_txt,
                file_name=f"{company}_{selected_subject}.txt",
                mime="text/plain"
            )
            
            # Add execution details OUTSIDE the status container
            with st.expander("Execution Details"):
                st.write("**Progress Log:**")
                for log_entry in progress_log:
                    st.write(log_entry)
                
                st.write(f"**Concurrency Settings:**")
                st.write(f"- Max concurrent sub-queries: {max_concurrent_subqueries}")
                st.write(f"- Max concurrent source queries: {max_concurrent_sources}")
        
        # Show error details OUTSIDE the status container
        if execution_error:
            st.error("An error occurred during processing")
            with st.expander("Error Details"):
                st.code(execution_error)
