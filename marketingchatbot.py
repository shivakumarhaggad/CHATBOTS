

import dspy
import google.generativeai as genai  # Add this line

# Configuring dspy settings with the base LM
gemini = dspy.Google(model='gemini-1.5-flash', api_key="AIzaSyAlu2qxavYR91heNy1xs4YJgtijqIrJYww", temperature=0.3)

dspy.settings.configure(lm=gemini)


# Define the MarketingChatbot class that inherits from dspy.Signature
class MarketingChatbot(dspy.Signature):
  """You are a Marketing Chatbot, whose main aim is to answer marketing queries of the user.
  You may also be given the history of the prompts and responses, use this history as the context while answering the query.
  All your responses should be strictly specific to marketing domain.
  
  If anything not related to marketing is given in the query, you have to politely refuse to answer.
  """

  # All the input and output variables, with their descriptions.
  history = dspy.InputField(desc="The history of prompts and responses")
  query = dspy.InputField(desc="The query of the user.")
  answer = dspy.OutputField(desc="The answer to the user's query.")

# The CoT class
class CoT(dspy.Module):

  # The constructor
  def __init__(self):
    super().__init__()
    self.program = dspy.ChainOfThought(MarketingChatbot)
  
  # The method used for calling the model.
  def forward(self, history, query):
    return self.program(history=history, query=query)

  
# Initializing the object of CoT class
marketingChatbot = CoT()

# Using the Chatbot -------

# To access the history of chats and responses
# Convert history from a list to a string
history = "\n".join(map(str, gemini.history))  # Converts list items to a string

# User's query
query = "I am building a free marketing chatbot, devise a marketing plan for it."

# Response from the Chatbot
response = marketingChatbot.forward(history=history, query=query)

# Printing the answer to the query from the response generated
print(response.answer)
