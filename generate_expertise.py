import os
import json
from datetime import datetime
from transformers import pipeline
import requests

# GitHub API Token (store securely in env variables)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("GITHUB_REPO")  

# GitHub API URL (Exclude PRs by filtering on pull_request field)
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_NAME}/issues?state=all&per_page=100&page={{}}"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# NLP Model for issue embedding
nlp_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

EXPERTISE_FILE = "expertise_data.json"

def generate_embedding(text):
    """Generate an embedding for issue text."""
    return nlp_model(text)[0][:4]  # Limiting to 4 dimensions for demo

def fetch_issues():
    """Fetch open and closed issues from GitHub API, excluding pull requests, and handling pagination."""
    expertise_data = {"contributors": {}}
    page = 1
    while True:
        response = requests.get(GITHUB_API_URL.format(page), headers=HEADERS)
        issues = response.json()
        if not issues:
            break  # Exit loop if no more issues
        
        for issue in issues:
            if "pull_request" in issue:
                continue  # Skip pull requests
            
            if issue.get("assignee"):
                contributor = issue["assignee"]["login"]
                issue_text = issue["title"] + " " + issue.get("body", "")
                embedding = generate_embedding(issue_text)
                timestamp = issue.get("closed_at", datetime.utcnow().isoformat())
                
                if contributor not in expertise_data["contributors"]:
                    expertise_data["contributors"][contributor] = {"issues": []}
                
                expertise_data["contributors"][contributor]["issues"].append({
                    "title": issue["title"],
                    "embedding": embedding,
                    "timestamp": timestamp
                })
        
        page += 1  # Move to the next page
    
    return expertise_data

def save_expertise_data(expertise_data):
    """Save expertise data to JSON file."""
    with open(EXPERTISE_FILE, "w") as f:
        json.dump(expertise_data, f, indent=4)

def main():
    expertise_data = fetch_issues()
    save_expertise_data(expertise_data)
    print(f"Expertise data saved to {EXPERTISE_FILE}")

if __name__ == "__main__":
    main()
