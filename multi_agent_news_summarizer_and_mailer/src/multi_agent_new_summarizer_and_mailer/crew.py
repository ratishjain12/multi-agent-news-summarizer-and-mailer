from crewai import Agent, Crew, Process, Task, LLM
from crewai.flow.flow import Flow, listen, start
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool,ScrapeWebsiteTool
from dotenv import load_dotenv
from pydantic import BaseModel
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import markdown
import os
import base64

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def authenticate_user():
    """
    Authenticate the user via OAuth2 and return the Gmail API service instance.
    """
    creds = None
    token_path = 'token.json'  # Token file to store user credentials
    
    # Check if token already exists
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If no valid credentials, prompt user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)  # Replace with your downloaded credentials file
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for future use
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

class EmailFlowState(BaseModel):
	data:str = ""

def send_email(subject: str, recipient_email: str, body: str):
    service = authenticate_user()

    # Convert Markdown to HTML
    html_body = markdown.markdown(body)
    
    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = "me"  # "me" is a special alias for the authenticated user
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))
    
    # Encode the message to base64
    raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    message = {'raw': raw_message}

    try:
        # Send the email
        sent_message = service.users().messages().send(userId="me", body=message).execute()
        print(f"Email sent successfully! Message ID: {sent_message['id']}")
    except Exception as e:
        print(f"An error occurred: {e}")

class EmailFlow(Flow[EmailFlowState]):
	@start()
	def generate_html(self):
		inputs = {
			"topic":"cricket",
			"recipient_email":"ratishjain10@gmail.com",
			"subject":"Latest news curated for you!!!"
		}
		result = MultiAgentNewsletterSummarizer().crew().kickoff(inputs=inputs)
		self.state.data = result.raw
	
	@listen(generate_html)
	def send_email(self):
		email_content = self.state.data
		recipient_email = "ratishjain10@gmail.com"  # You can extract this from state as well
		subject = "Latest News Summary: Cricket"
		send_email(subject = subject, recipient_email=recipient_email, body=email_content)




@CrewBase
class MultiAgentNewsletterSummarizer():
	"""MultiAgentNewsletterSummarizer crew"""

	llm=LLM(
		model="gemini/gemini-1.5-pro",
		api_key=os.environ['GOOGLE_API_KEY']
	)

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def news_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['news_researcher'],
			tools=[SerperDevTool()],
			llm=self.llm,
			verbose=True
		)

	@agent
	def news_scraper(self) -> Agent:
		return Agent(
			config=self.agents_config['news_scraper'],
			tools=[ScrapeWebsiteTool()],
			llm=self.llm,
			verbose=True
		)
	
	@agent
	def news_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['news_writer'],
			llm=self.llm,
			verbose=True
		)


	@task
	def news_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['news_research_task'],
		)	
	
	@task
	def news_scraper_task(self) -> Task:
		return Task(
			config=self.tasks_config['news_scraper_task'],
		)
	
	@task
	def news_writer_task(self) -> Task:
		return Task(
			config=self.tasks_config['news_writer_task'],
			output_file="report.md"
		)
	
	@crew
	def crew(self) -> Crew:
		"""Creates the MultiAgentNewsletterSummarizer crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
