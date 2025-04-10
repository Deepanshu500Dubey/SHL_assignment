from flask import Flask, render_template, request, jsonify
import pandas as pd
import faiss
import numpy as np
import re
import os
from flask import Response, json
from collections import OrderedDict
from collections import OrderedDict
import ast
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from typing import List, Optional, Dict, Any

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['JSON_SORT_KEYS'] = False  # Disable key sorting

GROQ_API_KEY = "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"

# Add this at the TOP LEVEL of your file
initialized = False  # ← Add this global flag

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Then add the before_request handler
@app.before_request
def initialize_app():
    global initialized, df_metadata, faiss_index, model, workers
    if not initialized:
        try:
            FAISS_INDEX_PATH = "precomputed_faiss_index.bin"
            METADATA_PATH = "metadata.parquet"
            MODEL_NAME = "paraphrase-MiniLM-L6-v2"

            # Add verification logging
            app.logger.info(f"Current directory: {os.getcwd()}")
            app.logger.info(f"File exists - FAISS index: {os.path.exists(FAISS_INDEX_PATH)}")
            app.logger.info(f"File exists - Metadata: {os.path.exists(METADATA_PATH)}")

            df_metadata = pd.read_parquet(METADATA_PATH)
            app.logger.info(f"Metadata loaded. Columns: {df_metadata.columns.tolist()}")

            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            app.logger.info("FAISS index loaded successfully")
            
            model = SentenceTransformer(MODEL_NAME)
            app.logger.info(f"Model loaded: {MODEL_NAME}")
            
            workers = initialize_workers()
            initialized = True
            
        except Exception as e:
            app.logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to initialize application resources")
# Keep all original class definitions and functions unchanged
# [Retain all Supervisor/Worker class code]
class Supervisor:
        def __init__(self, supervisor_name: str, supervisor_prompt: str, model: Any): # Indented this line
            self.name = supervisor_name
            self.prompt_template = supervisor_prompt
            self.model = model

        def format_prompt(self, team_members: List[str]) -> str: # Indented this line
            return self.prompt_template.format(team_members=", ".join(team_members))

class Worker:
        def __init__(self, worker_name: str, worker_prompt: str, supervisor: Supervisor, tools: Optional[List[Any]] = None): # Indented this line
            self.name = worker_name
            self.prompt_template = worker_prompt
            self.supervisor = supervisor
            self.tools = tools or []
            
        def clean_response(self, response: str) -> Any:  # Changed return type to Any # Indented this line
            # Extract content after last </think> tag
            if '</think>' in response:
                response = response.split('</think>')[-1]
            
            # Remove markdown formatting and numbering
            response = response.replace('**', '').strip()
            response = response.split(':')[-1].strip()
            
            # Custom cleaning per worker type
            if self.name == 'TestTypeAnalyst':
                return ''.join([c for c in response if c.isupper() or c == ','])
            elif self.name == 'Skill Extractor':
                return '\n'.join([s.split('. ')[-1] for s in response.split('\n')])
            elif self.name == 'Time Limit Identifier':
                return response.split()[0]
            elif self.name == 'Testing Type Identifier':
                # Special handling for testing type response
                response = response.strip('[]')
                parts = [part.strip().lower() for part in response.split(',')]
                return [part if part in ('yes', 'no') else 'no' for part in parts]
            
            return response.split('\n')[0].strip('"').strip()
        
        def process_input(self, user_input: str) -> str: # Added the missing process_input function # Indented this line
            prompt = f"{self.prompt_template}\n\nUser Input: {user_input}"
            messages = [HumanMessage(content=prompt)]
            response = self.supervisor.model.invoke(messages)
            return self.clean_response(response.content)


# [Keep initialize_workers() function]

def initialize_workers():
    groq_model = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0,
        streaming=True,
        api_key=GROQ_API_KEY
    )

    supervisor = Supervisor(
        supervisor_name="AssessmentCoordinator",
        supervisor_prompt="You manage these specialists: {team_members}. Coordinate assessment creation workflow. Select next worker strategically. FINISH when complete.",
        model=groq_model
    )

    # Define worker prompts inline for deployment safety
    return [
        Worker(
        worker_name="TestTypeAnalyst",
        worker_prompt='''You are an AI classifier that maps user inputs to test type codes from this taxonomy:

Test Types (Code: Description)

A: Ability & Aptitude (cognitive skills, problem-solving)

B: Biodata & Situational Judgement (past behavior, hypothetical scenarios)

C: Competencies (job-specific skills like leadership)

D: Development & 360 (growth feedback, multi-rater reviews)

E: Assessment Exercises (role-plays, case studies)

K: Knowledge & Skills (technical/domain expertise)

P: Personality & Behavior (traits, motivations)

S: Simulations (realistic job-task replicas)

Rules:

Return only the relevant letter codes (e.g., K, A,S).

Use commas for multiple matches (no spaces).

Prioritize specificity (e.g., "Python coding test" → K, not A).

Default to B for biographical/historical scenarios.

Examples:

Input: "Quiz on Java and cloud architecture" → K

Input: "Test how someone leads a team during a crisis" → C,S

Input: "Evaluate agreeableness and reaction to feedback" → P,D

Output Format:
Return only the letter code(s) as a comma-separated string (e.g., P or B,S).

''',
        supervisor=supervisor
    ),
    Worker(
        worker_name="Skill Extractor",
        worker_prompt='''You are a skill extractor for assessment design. Identify both hard and soft skills explicitly mentioned in the user’s input that are relevant to the test’s purpose.

Rules:
Focus: Extract hard skills (technical) and soft skills (non-technical):

✅ Hard Skills:

Tools: Python, SQL, AWS

Frameworks: TensorFlow, React

Domains: cybersecurity, CAD, data analysis

✅ Soft Skills:

communication, leadership, teamwork, problem-solving

🚫 Exclude:

Generic terms: "experience," "knowledge," "proficiency"

Job roles: "developer," "engineer"

Test Type Context: Use the test type code (A/B/C/D/E/K/P/S) to refine extraction:

Example: Test type K (Knowledge & Skills) → Prioritize hard skills like Python.

Example: Test type C (Competencies) → Include both hard skills (CAD) and soft skills (leadership).

Example: Test type P (Personality) → Extract only soft skills if mentioned (e.g., adaptability).

Normalization:

Standardize terms: JS → JavaScript, ML → machine learning.

Merge equivalents: CAD → Computer-Aided Design.

Output:

Return a comma-separated list (e.g., Python, leadership, CAD).

If no skills are found, return [].

Examples:
Input	Test Type	Output
“Test Python coding and teamwork.”	K	Python, teamwork
“Assess problem-solving and cloud architecture.”	A	problem-solving, cloud architecture
“Evaluate leadership and CAD proficiency.”	C	leadership, CAD
“Behavioral test focusing on communication.”	P	communication
“No skills mentioned.”	S	[]''',
        supervisor=supervisor
    ),
    Worker(
        worker_name="Job Level Identifier",
        worker_prompt='''You are an AI assistant tasked with identifying the job level for which a test is intended. Given input that may include job titles, responsibilities, or descriptions, determine the most appropriate job level from the following list:

Director

Entry Level

Executive

Frontline Manager

General Population

Graduate

Manager

Mid-Professional

Professional

Professional Individual Contributor

Supervisor

Use contextual clues in the input to make an accurate classification. Respond only with the job level.
 ''',
        supervisor=supervisor
    ),
    Worker(
        worker_name="Language Preference Identifier",
        worker_prompt='''You are a language detector for assessments. Identify spoken (natural) languages (e.g., English, Mandarin, Spanish) explicitly mentioned in the user’s input.

Rules:

Focus:

Extract only natural languages (e.g., "French", "Japanese").

Ignore programming languages (Python, Java), tools (SQL), or frameworks (React).

Defaults:

Return English if no spoken language is mentioned.

For multi-language requests (e.g., "English and Spanish"), return a comma-separated list: English, Spanish.

Output:

Use full language names (e.g., "German" not "Deutsch").

Case-insensitive (e.g., "spanish" → Spanish).

Examples:

Input: "Test must be in Portuguese." → Output: Portuguese

Input: "Python coding test with instructions in Arabic." → Output: Arabic

Input: "Math exam for Spanish-speaking students." → Output: Spanish

Input: "Timed Java assessment." → Output: English

Respond only with the language name(s). No explanations.

'''
,
        supervisor=supervisor
    ),
    Worker(
        worker_name="Time Limit Identifier",
        worker_prompt='''You are an AI that extracts explicit test durations from user input.

Rules
Extract:

Return exact phrases with a number + time unit (e.g., 90 minutes, 2.5 hrs, no more than 45 mins).

Include comparative phrasing (e.g., under 1 hour, at least 20 minutes).

Ignore:

Deadlines (e.g., submit by Friday).

Experience durations (e.g., 5 years of experience).

Vague terms (e.g., timed test, time-sensitive).

Output:

For valid durations: Return them as a comma-separated list (e.g., 1 hour, 30 mins).

For no valid durations: Return no time specified.

Examples
Input	Output
"Complete the test in 45 mins."	45 mins
"Section A: 1 hour; Section B: 30 mins."	1 hour, 30 mins
"Timed exam with no duration mentioned."	no time specified
"Submit by 5 PM and allow up to 2 hrs."	2 hrs
"Requires 3+ years of experience."	no time specified
Strict Constraints
Never return explanations, formatting, or placeholders.

Only return extracted durations or no time specified.

''',
        supervisor=supervisor
    ),
    Worker(
        worker_name="Testing Type Identifier",
        worker_prompt='''You are an AI classifier that detects mentions of remote testing or adaptive testing/IRT in user inputs and returns a structured response.

Rules
Detection Logic:

Remote Testing: yes if the exact phrase "remote testing" is present.

Adaptive Testing: yes if "adaptive testing" or "IRT" (case-insensitive) is present.

Default to no for missing terms.

Output Format:

Return [yes,yes] if both terms are present.

Return [yes,no] if only remote testing is mentioned.

Return [no,yes] if only adaptive testing/IRT is mentioned.

Return [no,no] if neither is mentioned.

Constraints:

NO explanations, NO deviations from the format.

Exact matches only (e.g., "remote" ≠ "remote testing").

Examples
Input	Output
"Conduct remote testing with IRT."	[yes,yes]
"Use adaptive testing."	[no,yes]
"Remote testing required."	[yes,no]
"Timed onsite exam."	[no,no]
Command:
Return ONLY the structured list ([yes,yes], [no,yes], etc.). No other text!''',
        supervisor=supervisor
    )
        # Add all other workers with their prompts here
        # ... [Keep other worker definitions unchanged] ...
    ]


import ast  # Make sure this is imported at the top of your file

# [Previous route setup code...]

@app.route('/recommend', methods=['POST'])
def api_recommendations():
    """API endpoint for JSON recommendations"""
    try:
        # Get JSON input
        data = request.get_json()
        user_input = data.get('query')
        
        if not user_input:
            return jsonify({"error": "Missing 'query' field"}), 400

        # Process input through workers
        input_list = process_user_input(user_input, workers)
        
        # Get recommendations
        recommendations = recommend_assessments(
            input_list, df_metadata, faiss_index, model
        )
        
        # Convert to required JSON format
        results = []
        if not recommendations.empty:
            for record in recommendations.to_dict('records'):
                # Safely handle test_type parsing
                test_type_value = record.get('test_type', '[]')
                try:
                    parsed_test_type = (ast.literal_eval(test_type_value) 
                                      if isinstance(test_type_value, str)
                                      else test_type_value)
                except (SyntaxError, ValueError, TypeError):
                    parsed_test_type = []
                
                results.append(OrderedDict([
                    ("url", record.get('URL', '')),
                    ("adaptive_support", "Yes" if str(record.get('Adaptive/IRT', 'No')).lower() == 'yes' else "No"),
                    ("duration", int(record.get('Assessment Length', 0))),
                    ("remote_support", "Yes" if str(record.get('Remote Testing', 'No')).lower() == 'yes' else "No"),
                    ("test_type", parsed_test_type),
                    ("description", record.get('Description', ''))
                ]))
        
        return Response(
            json.dumps({"recommended_assessments": results[:10]}, ensure_ascii=False),
            mimetype='application/json',
            status=200
        )
    
    except Exception as e:
        app.logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    
# [Keep recommend_assessments() function]
def recommend_assessments(input_list: List[str], 
                         df_metadata: pd.DataFrame,
                         faiss_index: faiss.Index,
                         model: SentenceTransformer) -> pd.DataFrame:
    """FAISS-powered recommendation engine"""
    # Input validation
    if len(input_list) < 6:
        return pd.DataFrame()

    try:
        # Unpack processed inputs
        remote_support, adaptive_support, test_type, skills_query, language, duration = input_list
        
        # Language matching
        def language_filter(lang_entry: str) -> bool:
            primary_lang = re.split(r'[^a-zA-Z]', language.lower())[0]
            return any(
                primary_lang == re.split(r'[^a-zA-Z]', l.strip())[0]
                for l in re.split(r'[,/]', str(lang_entry).lower())
            )

        # Test type matching
        def test_type_filter(test_entry: str) -> bool:
            query_terms = set(re.split(r'[\s-]+', test_type.lower()))
            entry_terms = set(re.split(r'[\s-]+', test_entry.lower()))
            return len(query_terms & entry_terms) > 0

        # Duration parsing
        def parse_duration(duration_str: str) -> Optional[int]:
            total = 0
            for match in re.finditer(
                r'(?P<value>\d+)\s*(?P<unit>mins?|minutes?|hrs?|hours?)?',
                duration_str,
                re.IGNORECASE
            ):
                value = int(match.group('value'))
                unit = (match.group('unit') or 'minutes').lower()
                total += value * 60 if unit.startswith('h') else value
            return total or None

        # Apply hard filters
        time_filter = parse_duration(duration)
        filtered = df_metadata[
            df_metadata['Language'].apply(language_filter) &
            df_metadata['Test Type'].apply(test_type_filter)
        ]
        
        if time_filter:
            filtered = filtered[filtered['Assessment Length'] <= time_filter]

        if filtered.empty:
            return pd.DataFrame()

        # FAISS semantic search
        query_embedding = model.encode([skills_query])[0].astype('float32')
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        filtered_indices = filtered.index.astype('int64').values
        k = min(50, len(filtered))
        
        distances, indices = faiss_index.search(
            query_embedding.reshape(1, -1), 
            k=k
        )
        
        # Map FAISS indices to metadata
        valid_matches = [
            (i, d) 
            for i, d in zip(indices[0], distances[0]) 
            if i in filtered_indices
        ]
        
        # Prepare results
        results = []
        for idx, distance in valid_matches:
            row = df_metadata.iloc[idx].copy()
            row['similarity'] = 1 - (distance / 4)  # L2 to similarity
            results.append(row)

        result_df = pd.DataFrame(results).sort_values('similarity', ascending=True)
        return result_df[['URL', 'Test Type', 'Remote Testing', 
                         'Adaptive/IRT', 'Assessment Length', "test_type","Individual Test Solutions",  
                         "Language", 
                         'Description', 'similarity']].head(10)  # Added URL and Description

    except Exception as e:
        app.logger.error(f"Recommendation error: {str(e)}")  # Log instead of st.error
        return pd.DataFrame()


# [Keep process_user_input() function]
def process_user_input(user_input: str, workers: List[Any]) -> list:
    """Process input through workers and format for recommendation system"""
    output_order = [
        "Testing Type Identifier",
        "TestTypeAnalyst",
        "Skill Extractor",
        "Job Level Identifier",
        "Language Preference Identifier",
        "Time Limit Identifier"
    ]
    
    # Get worker results
    results = {}
    for worker in workers:
        result = worker.process_input(user_input)
        results[worker.name] = result
    
    # Create ordered list
    original_list = [results[key] for key in output_order]
    
    # Transform to input_list format
    input_list = [
        *original_list[0],  # Flatten testing type identifiers
        original_list[1].replace(',', ' '),  # Test types
        f"{original_list[2]}, {original_list[3]}",  # Skills + Job Level
        *original_list[4:]  # Language and Duration
    ]
    
    return input_list

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            user_input = request.form.get('job_description', '')
            
            if not user_input:
                return render_template('index.html', 
                                    error="Please enter a job description first!")
            
            # Process input
            input_list = process_user_input(user_input, workers)
            
            
            # Get recommendations
            recommendations = recommend_assessments(
                input_list, df_metadata, faiss_index, model
            )
            
            # Prepare results
            results = recommendations[[
                'Individual Test Solutions', 'Test Type', 
                'Language', 'Remote Testing', 
                'Adaptive/IRT', 'Assessment Length'
            ]].to_dict('records') if not recommendations.empty else []
            
            return render_template('index.html', 
                                 results=results,
                                 input_text=user_input)
            
        except Exception as e:
            app.logger.error(f"Processing error: {str(e)}")
            return render_template('index.html', 
                                error=f"Error processing request: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)