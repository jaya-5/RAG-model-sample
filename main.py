import ollama

model = "gemma3:1b"

def ask_llm(inp):
    scrit_rules = f"""
You are a Financial Assistant.

Rules:
1. Help users understand finance concepts.
2. Explain investments, budgeting, savings, and loans.
3. Do not provide guaranteed financial advice.
4. Give educational information only.

User Input: {inp}
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": scrit_rules,
            }
        ]
    )

    print(response["message"]["content"])

while True:
    inp = input("Enter query: ")
    ask_llm(inp)
    
    