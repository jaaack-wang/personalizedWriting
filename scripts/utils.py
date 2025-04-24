import math
from litellm import completion
from nltk.tokenize import word_tokenize


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


def count_words(text):
    return len(word_tokenize(text))


def round_up_to_nearest_10(n):
    return math.ceil(n / 10) * 10


def list_writing_samples(samples):
    return '\n\n'.join([f'Sample#{ix+1}\n\n{sample.strip()}' 
                      for ix, sample in enumerate(samples)])


