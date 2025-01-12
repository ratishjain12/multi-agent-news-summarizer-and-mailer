from crewai import Agent, Crew, Process, Task, LLM
from crewai.flow.flow import Flow, listen, start
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool,ScrapeWebsiteTool
from dotenv import load_dotenv
from pydantic import BaseModel
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import markdown
import os

load_dotenv()


class EmailFlowState(BaseModel):
	data:str = ""

def send_email(subject: str, recipient_email: str, body: str):
    sender_email = os.getenv("EMAIL_SENDER")  # Your Gmail address
    sender_password = os.getenv("EMAIL_PASSWORD")  # Use the generated application-specific password

    # Convert Markdown to HTML
    html_body = markdown.markdown(body)  # Convert markdown to HTML

    # Prepare the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Attach the HTML content
    msg.attach(MIMEText(html_body, 'html'))  # Send as HTML

    try:
        # Send email via Gmail's SMTP server
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)  # Use the application-specific password
            server.send_message(msg)
            print(f"Email sent to {recipient_email} successfully!")

    except Exception as e:
        print(f"Error sending email: {str(e)}")

class EmailFlow(Flow[EmailFlowState]):
	@start()
	def generate_html(self):
		inputs = {
			"topic":"football",
			"recipient_email":"ratishjain10@gmail.com",
			"subject":"Latest news curated for you!!!"
		}
		result = MultiAgentNewsletterSummarizer().crew().kickoff(inputs=inputs)
		self.state.data = result.raw
	
	@listen(generate_html)
	def send_email(self):
		email_content = self.state.data
		recipient_email = "ratishjain10@gmail.com"  # You can extract this from state as well
		subject = "Latest News Summary: Football"
		send_email(subject = subject, recipient_email=recipient_email, body=email_content)




@CrewBase
class MultiAgentNewsletterSummarizer():
	"""MultiAgentNewsletterSummarizer crew"""

	llm=LLM(
		model="gemini/gemini-1.5-flash",
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
