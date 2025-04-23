from string import Template


def get_prompt_template_for_mimicking_writing():

    prompt_tmp = '''
You will be given one or more writing samples from a specific author. Your task is to analyze \
the author's style, tone, and voice, then craft a new piece that closely mimics their writing based on a provided summary.

### Author's Writing Sample(s)

$writing_samples

### Writing Task Summary

$summary

### Instructions

- Ensure your writing closely matches the provided samples in terms of tone, vocabulary, sentence structure, and overall style.
- Maintain consistency with the author's voice while accurately reflecting the details of the given summary.
- Strive to make your writing indistinguishable from the original author's work.

Begin your response below:
    '''.strip()

    return Template(prompt_tmp)


def get_prompt_template_for_summarizing_text():
    
    prompt_tmp = '''
You will be given a piece of text. Your task is to summarize the text in a concise and clear manner, capturing the main ideas and key points while maintaining the original meaning.


### Text to Summarize


$text


### Instructions


- Provide a summary that is brief yet comprehensive.
- Ensure that the summary accurately reflects the content of the original text.
- Avoid adding any personal opinions or interpretations.
- Do not output anything other than the summary.


Begin your response below:
    '''.strip()

    return Template(prompt_tmp)


