from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import json
import markdown
import tiktoken
import os
from openai import OpenAI

# =============================================================================
# TOKEN COUNTING UTILITIES
# =============================================================================

def count_tokens(messages, model="gpt-4o-mini"):
    """
    Count tokens in a message list.
    Works for text and estimates for images.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    
    for message in messages:
        # Count tokens for role
        total_tokens += 4  # every message has role overhead
        
        if isinstance(message.get("content"), str):
            # Simple text content
            total_tokens += len(encoding.encode(message["content"]))
        
        elif isinstance(message.get("content"), list):
            # Complex content (text + images)
            for content_part in message["content"]:
                if content_part.get("type") == "text":
                    total_tokens += len(encoding.encode(content_part["text"]))
                
                elif content_part.get("type") == "image_url":
                    # Image token estimation
                    total_tokens += estimate_image_tokens(content_part)
    
    total_tokens += 2  # every reply is primed with assistant message
    
    return total_tokens


def estimate_image_tokens(image_content):
    """
    Estimate tokens for an image.
    OpenAI charges ~85-170 tokens per image depending on detail level.
    """
    detail = image_content.get("image_url", {}).get("detail", "auto")
    
    if detail == "low":
        return 85
    else:
        # High detail: 85 base + 170 per 512x512 tile
        # For safety, estimate 255 tokens (assumes ~1 tile)
        return 255

# =============================================================================
# FLASK APP INITIALIZATION
# =============================================================================

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['clinical_trials']
collection = db['studies']

# OpenAI setup
with open('data1/key/openai_key.txt') as f:
    openai_key = f.readline().strip()

os.environ["OPENAI_API_KEY"] = openai_key
openai_client = OpenAI()

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/interventions', methods=['GET'])
def get_interventions():
    """Get list of unique interventions from database"""
    try:
        # Aggregate to get unique intervention names
        # Works with simplified data structure: {interventions: [string1, string2, ...]}
        pipeline = [
            {'$unwind': '$interventions'},
            {'$group': {'_id': '$interventions'}},
            {'$sort': {'_id': 1}},
            {'$limit': 500}  # Limit to most common 500
        ]
        
        interventions = collection.aggregate(pipeline)
        intervention_list = [doc['_id'] for doc in interventions if doc['_id']]
        
        return jsonify({'interventions': intervention_list})
    except Exception as e:
        print(f"Error fetching interventions: {str(e)}")
        return jsonify({'interventions': []})


@app.route('/api/search', methods=['POST'])
def search_studies():
    """Search for clinical trials based on filters"""
    filters = request.json
    query = build_query_from_filters(filters)
    
    page = filters.get('page', 1)
    per_page = min(filters.get('per_page', 20), 100)
    skip = (page - 1) * per_page
    
    total = collection.count_documents(query)
    results = list(collection.find(query).skip(skip).limit(per_page))
    
    for result in results:
        result['_id'] = str(result['_id'])
    
    return jsonify({
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
        'results': results
    })


@app.route('/api/study/<nct_id>', methods=['GET'])
def get_study(nct_id):
    """Get a specific study by NCT ID"""
    study = collection.find_one({
        'protocolSection.identificationModule.nctId': nct_id
    })
    
    if study:
        study['_id'] = str(study['_id'])
        return jsonify(study)
    return jsonify({'error': 'Study not found'}), 404


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for individual study"""
    data = request.json
    nct_id = data.get('nctId')
    question = data.get('question')
    
    study = collection.find_one({'protocolSection.identificationModule.nctId': nct_id})
    if not study:
        return jsonify({'error': 'Study not found'}), 404
    
    study_copy = {k: v for k, v in study.items() if k != '_id'}
    study_context = json.dumps(study_copy, indent=2)
    
    estimated_tokens = len(study_context) / 4
    max_context_tokens = 6000
    if estimated_tokens > max_context_tokens:
        max_chars = max_context_tokens * 4
        study_context = study_context[:max_chars] + "\n\n[Context truncated due to length]"
    
    system_message = f"""You are a clinical trials expert. Answer questions about this study based on the data below.

STUDY DATA:
{study_context}

Instructions:
- Answer based only on the provided data
- If information isn't in the data, say so
- Be precise and reference specific fields when relevant
- Keep answers concise but complete"""
    
    try:
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        token_count = count_tokens(message_list, model="gpt-4o-mini")
        print(f"📊 Token count: {token_count:,} tokens")
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_list,
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = completion.choices[0].message.content
        answer_html = markdown.markdown(answer, extensions=['extra', 'nl2br'])
        
        return jsonify({'answer': answer_html})
        
    except Exception as e:
        return jsonify({'error': f'AI error: {str(e)}'}), 500


@app.route('/api/chat-all', methods=['POST'])
def chat_all():
    """Chat endpoint for all filtered studies"""
    data = request.json
    filters = data.get('filters', {})
    question = data.get('question', '')
    advanced_mode = data.get('advancedMode', False)
    
    # Build query from filters
    query = build_query_from_filters(filters)
    
    # Get total count
    total_count = collection.count_documents(query)
    
    # Different limits based on mode
    if advanced_mode:
        max_studies = 100
    else:
        max_studies = 500
    
    limit = min(total_count, max_studies)
    studies = list(collection.find(query).limit(limit))
    
    # Process based on mode
    if advanced_mode:
        # Use COMPLETE data - just remove _id
        processed_studies = []
        for study in studies:
            study_copy = {k: v for k, v in study.items() if k != '_id'}
            processed_studies.append(study_copy)
        mode_info = f"Analyzing {len(processed_studies)} studies with COMPLETE data"
    else:
        # Use essential fields only
        processed_studies = [extract_essential_fields(study) for study in studies]
        mode_info = f"Analyzing {len(processed_studies)} studies with essential data"
    
    # Convert to JSON
    studies_json = json.dumps(processed_studies, indent=1)
    
    # Estimate tokens and handle limits
    estimated_tokens = len(studies_json) / 4
    max_context_tokens = 120000 if advanced_mode else 100000
    
    truncated_message = ""
    if estimated_tokens > max_context_tokens:
        chars_per_study = len(studies_json) / len(processed_studies)
        max_studies_fit = int(max_context_tokens * 4 / chars_per_study)
        
        if max_studies_fit < 1:
            return jsonify({
                'error': f'Studies are too large for {("advanced" if advanced_mode else "essential")} mode. Try {"essential mode or " if advanced_mode else ""}fewer studies.'
            }), 400
        
        processed_studies = processed_studies[:max_studies_fit]
        studies_json = json.dumps(processed_studies, indent=1)
        truncated_message = f"\n\n[Note: Showing {len(processed_studies)} out of {total_count} studies due to context limits]"
        mode_info += f" (truncated to {len(processed_studies)} studies)"
    
    system_message = f"""You are a clinical trials research analyst. Answer questions about this collection of clinical trial studies.

DATASET SUMMARY:
- Total studies in filtered results: {total_count}
- Studies provided for analysis: {len(processed_studies)}
- Data mode: {"COMPLETE (all fields)" if advanced_mode else "ESSENTIAL (key fields only)"}

STUDY DATA:
{studies_json}{truncated_message}

Instructions:
- Analyze the provided studies to answer the question
- Provide statistics, trends, and insights when relevant
- If asked about specific studies, cite NCT IDs
- If information spans many studies, provide summaries and key patterns
- Be precise and data-driven
- Use markdown formatting for readability
{"- You have access to COMPLETE study data including descriptions, outcomes, eligibility criteria, etc." if advanced_mode else "- You have access to essential study data. For detailed information about specific studies, suggest using the individual study chat."}"""
    
    try:
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        # Count tokens for logging
        token_count = count_tokens(message_list, model="gpt-4o")
        print(f"📊 Token count for chat-all ({mode_info}): {token_count:,} tokens")
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_list,
            temperature=0.3,
            max_tokens=2000
        )
        
        answer = completion.choices[0].message.content
        answer_html = markdown.markdown(answer, extensions=['extra', 'nl2br', 'tables'])
        
        return jsonify({
            'answer': answer_html,
            'info': mode_info
        })
        
    except Exception as e:
        print(f"Error in chat-all: {str(e)}")
        return jsonify({'error': f'AI error: {str(e)}'}), 500

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_query_from_filters(filters):
    """Build MongoDB query from filters"""
    query = {}
    
    if filters.get('condition'):
        query['protocolSection.conditionsModule.conditions'] = {
            '$regex': filters['condition'], '$options': 'i'
        }
    
    if filters.get('intervention') and len(filters['intervention']) > 0:
        query['protocolSection.armsInterventionsModule.interventions.name'] = {
            '$in': filters['intervention']
        }
    
    if filters.get('status') and len(filters['status']) > 0:
        query['protocolSection.statusModule.overallStatus'] = {'$in': filters['status']}
    
    if filters.get('studyType') and len(filters['studyType']) > 0:
        query['protocolSection.designModule.studyType'] = {'$in': filters['studyType']}
    
    if filters.get('phase') and len(filters['phase']) > 0:
        query['protocolSection.designModule.phases'] = {
            '$elemMatch': {'$in': filters['phase']}
        }
    
    if filters.get('sex') and filters['sex'] != 'ALL':
        query['protocolSection.eligibilityModule.sex'] = filters['sex']
    
    if filters.get('ageGroups') and len(filters['ageGroups']) > 0:
        query['protocolSection.eligibilityModule.stdAges'] = {
            '$elemMatch': {'$in': filters['ageGroups']}
        }
    
    if filters.get('healthyVolunteers'):
        query['protocolSection.eligibilityModule.healthyVolunteers'] = True
    
    if filters.get('hasResults') == 'true':
        query['hasResults'] = True
    elif filters.get('hasResults') == 'false':
        query['hasResults'] = False
    
    if filters.get('hasProtocol'):
        query['documentSection.largeDocumentModule.largeDocs.hasProtocol'] = True
    if filters.get('hasSAP'):
        query['documentSection.largeDocumentModule.largeDocs.hasSap'] = True
    if filters.get('hasICF'):
        query['documentSection.largeDocumentModule.largeDocs.hasIcf'] = True
    
    if filters.get('funderType') and len(filters['funderType']) > 0:
        query['protocolSection.sponsorCollaboratorsModule.leadSponsor.class'] = {
            '$in': filters['funderType']
        }
    
    if filters.get('location'):
        location_regex = {'$regex': filters['location'], '$options': 'i'}
        query['$or'] = [
            {'protocolSection.contactsLocationsModule.locations.country': location_regex},
            {'protocolSection.contactsLocationsModule.locations.city': location_regex},
            {'protocolSection.contactsLocationsModule.locations.state': location_regex}
        ]
    
    date_fields = {
        'studyStart': 'protocolSection.statusModule.startDateStruct.date',
        'primaryCompletion': 'protocolSection.statusModule.primaryCompletionDateStruct.date',
    }
    
    for filter_key, field_path in date_fields.items():
        from_date = filters.get(f'{filter_key}From')
        to_date = filters.get(f'{filter_key}To')
        
        if from_date or to_date:
            query[field_path] = {}
            if from_date:
                query[field_path]['$gte'] = from_date
            if to_date:
                query[field_path]['$lte'] = to_date
    
    if filters.get('title'):
        title_regex = {'$regex': filters['title'], '$options': 'i'}
        if '$or' not in query:
            query['$or'] = []
        query['$or'].extend([
            {'protocolSection.identificationModule.briefTitle': title_regex},
            {'protocolSection.identificationModule.officialTitle': title_regex},
            {'protocolSection.identificationModule.acronym': title_regex}
        ])
    
    if filters.get('sponsor'):
        query['protocolSection.sponsorCollaboratorsModule.leadSponsor.name'] = {
            '$regex': filters['sponsor'], '$options': 'i'
        }
    
    if filters.get('outcome'):
        outcome_regex = {'$regex': filters['outcome'], '$options': 'i'}
        if '$or' not in query:
            query['$or'] = []
        query['$or'].extend([
            {'protocolSection.outcomesModule.primaryOutcomes.measure': outcome_regex},
            {'protocolSection.outcomesModule.secondaryOutcomes.measure': outcome_regex}
        ])
    
    if filters.get('nctId'):
        query['protocolSection.identificationModule.nctId'] = filters['nctId'].upper()
    
    if filters.get('fdaaa801Violation'):
        query['protocolSection.oversightModule.fdaaa801Violation'] = True
    
    return query


def extract_essential_fields(study):
    """Extract only essential fields from a study to reduce token usage"""
    ps = study.get('protocolSection', {})
    
    # Identification
    ident = ps.get('identificationModule', {})
    nct_id = ident.get('nctId', 'N/A')
    title = ident.get('briefTitle', 'N/A')
    
    # Status
    status_mod = ps.get('statusModule', {})
    status = status_mod.get('overallStatus', 'N/A')
    
    # Design
    design = ps.get('designModule', {})
    study_type = design.get('studyType', 'N/A')
    phases = design.get('phases', [])
    
    # Conditions
    conditions = ps.get('conditionsModule', {}).get('conditions', [])
    
    # Interventions
    interventions_list = ps.get('armsInterventionsModule', {}).get('interventions', [])
    interventions = [i.get('name', '') for i in interventions_list[:3]]  # Limit to first 3
    
    # Outcomes - just the first primary outcome
    outcomes = ps.get('outcomesModule', {})
    primary_outcomes = outcomes.get('primaryOutcomes', [])
    primary_outcome = primary_outcomes[0].get('measure', 'N/A') if primary_outcomes else 'N/A'
    
    # Sponsor
    sponsor_mod = ps.get('sponsorCollaboratorsModule', {})
    sponsor = sponsor_mod.get('leadSponsor', {}).get('name', 'N/A')
    
    # Enrollment
    design_mod = ps.get('designModule', {})
    enrollment = design_mod.get('enrollmentInfo', {}).get('count', 'N/A')
    
    # Eligibility
    eligibility = ps.get('eligibilityModule', {})
    sex = eligibility.get('sex', 'N/A')
    min_age = eligibility.get('minimumAge', 'N/A')
    max_age = eligibility.get('maximumAge', 'N/A')
    
    # Locations - just countries
    locations = ps.get('contactsLocationsModule', {}).get('locations', [])
    countries = list(set([loc.get('country', '') for loc in locations if loc.get('country')]))[:5]
    
    return {
        'nctId': nct_id,
        'title': title,
        'status': status,
        'studyType': study_type,
        'phases': phases,
        'conditions': conditions[:5],  # Limit to 5
        'interventions': interventions,
        'primaryOutcome': primary_outcome,
        'sponsor': sponsor,
        'enrollment': enrollment,
        'eligibility': {
            'sex': sex,
            'ageRange': f"{min_age} to {max_age}"
        },
        'countries': countries
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Clinical Trials Search Application")
    print("=" * 60)
    print("Starting server...")
    print("Open your browser and go to: http://localhost:5033")
    print("=" * 60)
    app.run(debug=True, port=5033)

