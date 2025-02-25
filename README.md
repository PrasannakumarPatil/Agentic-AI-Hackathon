# Agentic-AI-Hackathon

![image](https://github.com/user-attachments/assets/6f7d2802-359d-4dc4-bbed-bc3ebbcd9164)


# Flow summary of the developed solution 

- A user creates an issue on GitHub
- Code Engine which is periodically scheduled fetched the unassigned issues from GitHub
- AI powered issue assignment  analyze the issue and classify it
- The Expert traction system stores contributors' embeddings 
- AI agents determine the best person to assign the issue, based on the issue classifier and expert agent 
- As the expert agent is finds the best match, the system comment notify the contributor on the GitHub 


### **GitHub Issue Auto-Assignment with Multi-Agent AI**  

This project automates GitHub issue assignment using a multi-agent AI system. It analyzes issue descriptions, generates embeddings, and assigns the most suitable contributor based on expertise tracking.  

## **Features**  
- Fetches unassigned GitHub issues  
- Uses embeddings to match issues with contributors  
- Suggests the best-fit contributor for each issue  
- Adds a comment on the issue with the recommendation  
- Periodic execution via Watsonx  

---

## **Setup Guide**  

### **1. Clone the Repository**  
```sh
git clone https://github.com/PrasannakumarPatil/Agentic-AI-Hackathon.git
cd Agentic-AI-Hackathon.git
```

### **2. Create a Python Virtual Environment**  
```sh
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3. Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4. Generate the expertise json file**
generate the expertise json file with the below command  
```sh
export GITHUB_TOKEN=<your github token>
export GITHUB_REPO=<your github repo>
python3 generate_expertise.py
```

### **5. Add this generated json file in WatsonX project data asset**

### **6. Set Up Environment Variables in WatsonX job**   
In the WatsonX project's job configuration add the following env. variables 
```sh
GITHUB_TOKEN=your_github_token
REPO_OWNER=your_repo_owner
REPO_NAME=your_repo_name
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id
IBM_COS_ENDPOINT=https://s3.direct.us-south.cloud-object-storage.appdomain.cloud
IBM_COS_API_KEY=your_cos_api_key
IBM_COS_BUCKET=your_bucket_name
EXPERTISE_FILE=expertise_data.json
```

### **7. Run the Job in WatsonX**  

---

## **Running as a Periodic Job in Watsonx**  

1. **Use Watsonx Scheduler**  
   - Open Watsonx and navigate to **Jobs > Create Job**  
   - Set the script path and environment  
   - Schedule execution at regular intervals  

---

## **Troubleshooting**  
- Ensure Python 3.11+ is installed  
- Verify API keys and repository access  
- Check IBM Cloud credentials for Object Storage access  

---
