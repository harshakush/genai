from ollama import Client

# Connect to Ollama
ollama = Client()

# Past issues
past_issues = [
    "App crashes on login with error code 504.",
    "Unable to reset password due to missing email link.",
    "Payment failed during checkout with Visa card.",
    "Error loading dashboard after latest update.",
    "Push notifications not received on Android devices."
]

# New customer query
query = "Customer gets 504 error when trying to log in."

# Format prompt with full issue list
context = "\n".join(f"- {issue}" for issue in past_issues)
prompt = f"""
You are a support assistant. Here is a list of known customer-reported issues:

{context}

Now, a new issue has been reported:
"{query}"

From the list above, identify the 2 or 3 most similar or relevant issues to this new query. Just return the matching issues verbatim.
"""

# Ask phi1_pavan to retrieve
response = ollama.chat(model="phi1_pavan:latest", messages=[
    {"role": "user", "content": prompt}
])

print("\nLLM-selected relevant issues:")
print(response['message']['content'])
