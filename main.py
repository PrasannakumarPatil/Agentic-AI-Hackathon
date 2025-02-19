import os
import json
import torch
import requests
import ibm_boto3
from botocore.client import Config
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from transformers import AutoTokenizer, AutoModel
import numpy as np

os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WATSONX_PROJECT_ID"] = "WatsonX Project ID"


GITHUB_TOKEN = ""  # Replace with a valid GitHub token
REPO_OWNER = ""  # Replace with the repository owner
REPO_NAME = ""  # Replace with the repository name
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
# IBM COS Credentials
COS_API_KEY_ID = ''
COS_AUTH_ENDPOINT = ""
COS_RESOURCE_ENDPOINT = ''
BUCKET_NAME = ''
OBJECT_KEY = 'expertise_data.json'

# Initialize IBM COS Client
cos_client = ibm_boto3.client(
    service_name='s3',
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version='oauth'),
    endpoint_url=COS_RESOURCE_ENDPOINT
)

def load_expertise_data():
    try:
        response = cos_client.get_object(Bucket=BUCKET_NAME, Key=OBJECT_KEY)
        data = json.loads(response['Body'].read().decode('utf-8'))
        print("Successfully loaded expertise data.")
        return data
    except Exception as e:
        print(f"Error loading expertise data: {e}")
        return None

# Load expertise data
EXPERTISE_FILE = load_expertise_data()

llm = LLM(
    model="watsonx/meta-llama/llama-3-1-70b-instruct",
    api_key="",
    temperature=0.7,    # Adjust based on task
    max_tokens=4096,    # Set based on output needs
)

# Load tokenizer and model for embedding generation
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModel.from_pretrained(MODEL_NAME)

EMBEDDING_DIM = 384  # Ensure embeddings are always this size

def generate_embedding(text):
    """Generate an embedding for the given text using AutoModel."""
    if not text:
        return np.zeros(EMBEDDING_DIM).tolist()  # Return zero vector if text is None
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.tolist()


# # Custom Tool to Generate Embeddings
class EmbeddingTool(BaseTool):
    name: str = "Embedding Tool"
    description: str = "Generates embeddings for a given text using transformers AutoModel."

    def _run(self, specific_text: str) -> list:
        """Generate embeddings for the given text."""
        return generate_embedding(specific_text)

# Agent that Executes the Tool
embedding_agent = Agent(
    role="Embedding Generator",
    goal="Generate embeddings for a given string - {specific_text}",
    backstory="An AI agent that generates embeddings using transformers.",
    verbose=True,
    tools=[EmbeddingTool()],
    llm=llm,
)

embedding_task = Task(
    description='Generate embeddings for a given string {specific_text} pass them to another agent.',
    expected_output="A list of embeddings.",
    agent=embedding_agent,
    tools=[EmbeddingTool()],
    #input=specific_text
)

# Contributor Suggestion Tool
class ContributorSuggestionTool(BaseTool):
    name: str = "Contributor Suggestion Tool"
    description: str = "Suggests the best contributor based on issue embedding similarity."

    def _run(self, issue_embedding: list) -> str:
        """Suggest the best contributor based on cosine similarity."""
        if not EXPERTISE_FILE:
            return "No expertise data available."
        
        best_match = None
        best_score = float("-inf")

        issue_embedding_tensor = torch.tensor(issue_embedding, dtype=torch.float32)
        
        for contributor, data in EXPERTISE_FILE.get("contributors", {}).items():
            past_embeddings = [torch.tensor(past_issue["embedding"], dtype=torch.float32) for past_issue in data.get("issues", [])]
            
            if not past_embeddings:
                continue
            
            past_embeddings_tensor = torch.stack(past_embeddings)
            similarities = torch.nn.functional.cosine_similarity(issue_embedding_tensor.unsqueeze(0), past_embeddings_tensor, dim=1)
            max_similarity = similarities.max().item()
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = contributor

        return best_match if best_match else "No suitable contributor found."

# Agent for Contributor Suggestion
suggestion_agent = Agent(
    role="Contributor Matcher",
    goal="Identify the best contributor for a given issue.",
    backstory="An AI agent that suggests the best contributor based on issue embeddings.",
    verbose=True,
    tools=[ContributorSuggestionTool()],
    llm=llm,
)

# Task for Contributor Suggestion
suggestion_task = Task(
    description="Identify the best contributor for an issue based on embeddings",
    expected_output="The best matching contributor.",
    agent=suggestion_agent,
    tools=[ContributorSuggestionTool()],
    input={"issue_embedding": embedding_task}
)

# Comment on GitHub Issue
def comment_on_issue(issue_number, comment):
    url = f"https://api.github.ibm.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"body": comment}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print(f"Commented on issue #{issue_number}")
    else:
        print(f"Failed to comment on issue #{issue_number}: {response.text}")


# Fetch unassigned GitHub issues
def fetch_unassigned_issues():
    url = f"https://api.github.ibm.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("Failed to fetch issues:", response.text)
        return []
    
    issues = response.json()
    return [issue for issue in issues if not issue.get("assignee")]

# Process each unassigned issue
def process_issues():
    unassigned_issues = fetch_unassigned_issues()
    
    if not unassigned_issues:
        print("No unassigned issues found.")
        return
    
    crew = Crew(agents=[embedding_agent, suggestion_agent], tasks=[embedding_task, suggestion_task])
    
    for issue in unassigned_issues:
        if "pull_request" in issue:
            continue
        issue_text = issue["title"] + " " + issue.get("body", "")
        result = crew.kickoff(inputs={"specific_text": issue_text})
        print(f"Issue: {issue['title']} -> Suggested Contributor: {result}")

        if result and result != "No suitable contributor found." and result != "No expertise data available.":
            comment = f"@{result}, you seem to be the best fit for this issue. Could you take a look?"
            comment_on_issue(issue["number"], comment)

# Run the process
process_issues()
