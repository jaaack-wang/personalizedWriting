from litellm import completion


def get_completion(prompt, 
                   model="openai/gpt-4.1-mini-2025-04-14", 
                   temperature=0, max_tries=5):
    
    for _ in range(max_tries):
        try:
            # Call the completion function with the provided parameters
            response = completion(
                model=model, temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["choices"][0]["message"]["content"]
        
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return "SOMETHING_WRONG"