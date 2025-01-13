#!/usr/bin/env python
import warnings
from crew import EmailFlow
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    EmailFlow().kickoff()
    

run()