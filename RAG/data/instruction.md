## 1. Overview
A **Python** script that reads a list of words and their parts of speech from a CSV, uses the OpenAI API to generate for each:

- **definition** (the positive)  
- **close_definition** (hard negative)  
- **sentence** (hard negative)  

and writes out a single JSON file containing an array of these records.

## 2. Inputs
- **CSV file** with two columns, **word** and **part_of_speech**.

## 3. Output
- A **.json** file containing an **array** of objects, each with:
  ```json
  {
    "word": "<input word>",
    "part_of_speech": "<input POS>",
    "definition": "<precise definition>",
    "close_definition": "<wrong but semantically close definition>",
    "sentence": "<example sentence using the word>"
  }
  ```

## 4. Configuration
All parameters are defined as globals at the top of the script:
- `SOURCE_CSV_PATH`: path to the input CSV file  
- `OUTPUT_JSON_PATH`: path where the resulting JSON array will be written  
- `PROMPT_TEMPLATE`: a Python string with placeholders `{word}` and `{part_of_speech}`  
- OpenAI credentials are read from environment variables (`OPENAI_API_KEY`)

## 5. High-Level Steps
1. **Read** the CSV (e.g. with `csv` or `pandas`).  
2. **For each** row:
   - Fill in `PROMPT_TEMPLATE`.  
   - Call the OpenAI API.  
   - **Parse** the response as JSON.  
   - **Validate** it has all three fields.  
     - If parsing/validation fails, **retry** automatically (use `@retry` from the `retry` library).  
3. **Collect** each valid record into a list.  
4. **Write** the list to `OUTPUT_JSON_PATH` as a JSON array.

## 6. Template

```
You are a helpful lexicographer.
You have to provide a definition, a close definition, and an example sentence for the given word.
1. Definition (concise, dictionary style).
2. Close definition (wrong but semantically similar; hard negative).
3. Example sentence using the word (hard negative).

Respond only as valid JSON:
{{
"definition": "...",
"close_definition": "...",
"sentence": "..."
}}

# Example
Word: "bank"
Part of speech: "noun"
Output:
```json
{{
"definition": "A financial institution that accepts deposits from the public.",
"close_definition": "A financial institution responsible for managing the country's currency.",
"sentence": "I went to the bank to withdraw some cash."
}}
```

# Task
Word: "{word}"
Part of speech: "{part_of_speech}"
Output:
```
