# AIT 526 - Programming Assignment 2
# Purpose: A simple question-answering system that uses Wikipedia to answer questions.
# Team 12
# Members: Aakiff Panjwani, Vansh Setpal, Hamaad Zuberi

import wikipedia
import sys
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import tree2conlltags

# Download necessary NLTK datasets; these may be cached locally on subsequent runs.
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

def parse_question(question):
    # Remove extra spaces and trailing punctuation.
    q = question.strip().rstrip('.!?')
    # Define regex pattern to extract type-of-question, subject, and an optional keyword.
    pattern = re.compile(
        r'^(Who|What|When|Where)\s+(?:is|was)\s+(.+?)(?:\s+(born|birthday|age))?$',
        re.IGNORECASE
    )
    # Attempt to match the question against the regex.
    match = pattern.match(q)
    if match:
        # qtype is 'who', 'what', 'when', or 'where'.
        qtype = match.group(1).lower()
        # subject indicates the entity the question is about.
        subject = match.group(2).strip()
        # keyword might be a detail like 'born', 'birthday', or 'age'.
        keyword = match.group(3).lower() if match.group(3) else None
        return qtype, subject, keyword
    # If no match, return the full question as subject.
    return None, question, None

def token(text):
    # Tokenize the text into words.
    tokens = word_tokenize(text)
    # Tag each token with its part-of-speech.
    tagged = pos_tag(tokens)
    # Build and return a named entity chunk tree in CONLL format.
    tree = ne_chunk(tagged)
    return tree2conlltags(tree)

def get_summary(subject):
    # Attempts to get a one-sentence summary from Wikipedia for the subject.
    try:
        return wikipedia.summary(subject, sentences=1)
    except Exception:
        # If an error occurs (e.g., no page found), return an empty string.
        return ""

def extract_date(text):
    # Define several date patterns:
    # 1. Day Month Year format, e.g., "22 February 1732"
    # 2. Month Day, Year format, e.g., "February 22, 1732"
    # 3. Fallback to a four-digit year.
    patterns = [
        r'\b\d{1,2} [A-Z][a-z]+ \d{4}\b',
        r'\b[A-Z][a-z]+ \d{1,2}, \d{4}\b',
        r'\b\d{4}\b'
    ]
    # Search each pattern in the provided text.
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            # Exclude false matches that might include unwanted text.
            if "to" not in m and "-" not in m:
                return m
    # Return empty string if no valid date is found.
    return ""

def extract_location(summary, subject):
    # Convert both summary and subject to lowercase to improve match reliability.
    summary = summary.lower()
    subject_lower = subject.lower()

    # Define regex patterns that try to locate phrases indicating a location.
    patterns = [
        rf"{subject_lower}.*?was born in ([a-zA-Z ,\-]+)",
        rf"{subject_lower}.*?is located in ([a-zA-Z ,\-]+)",
        rf"{subject_lower}.*?is situated in ([a-zA-Z ,\-]+)",
        rf"{subject_lower}.*?is in ([a-zA-Z ,\-]+)"
    ]

    # More general regex capturing a location following common prepositions.
    prepositions = r"(in|at|near|on|from|by|to)"
    location_pattern = rf"\b{prepositions}\s+((?:[A-Za-z0-9\s,\.\-\'])+)"

    # Search using the general location pattern.
    match = re.search(location_pattern, summary)
    if match:
        return match.group(2).strip()

    # Return an empty string if no location information is found.
    return ""

def format_answer(qtype, subject, keyword, info):
    # If no information was found, reply with a default error message.
    if not info:
        return "I am sorry, I don't know the answer."
    
    # Handle questions that start with "When".
    if qtype == "when":
        # If the question pertains to age calculation.
        if keyword == "age":
            year_match = re.search(r'\d{4}', info)
            if year_match:
                # Calculate age based on the current year and the extracted year.
                age = datetime.now().year - int(year_match.group())
                return f"{subject}'s age is {age}."
            else:
                return "I am sorry, I don't know the answer."
        # If the question is about birth date or birthday.
        elif keyword in ["born", "birthday"]:
            return f"{subject} was born on {info}."
        else:
            return f"{subject} is associated with {info}."
    # Handle "Where" questions.
    elif qtype == "where":
        return f"{subject} is located in {info}."
    # Handle "Who" and "What" questions.
    elif qtype in ["who", "what"]:
        # Ensure the answer ends with a period.
        return info if info.endswith('.') else info + '.'
    else:
        return info

def main():
    log = None
    try:
        # Ensure a log file is provided; add a default if none given.
        if len(sys.argv) < 2:
            sys.argv.append("mylogfile.txt")
        # Open the log file to record the session.
        log = open(sys.argv[1], "a+", encoding="utf-8")
        # Log the program start time.
        log.write(f"--- Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
        log.flush()
        
        # Display welcome messages to the user.
        print("This is a QA system")
        print("It will try to answer questions that start with Who, What, When or Where.")
        print("Enter 'exit' to leave the program.\n")
        counter = 1
        
        # Main loop to continuously accept questions from the user.
        while True:
            # Prompt the user to enter a question.
            user_input = input("Shoot me a question?\n").strip()
            # If the user types "exit", end the program.
            if user_input.lower() == "exit":
                print("Thank you! Goodbye.")
                log.write("User exited the program.\n")
                log.flush()
                break
            # Log the user's question with a counter.
            log.write(f"{counter} Q) {user_input}\n")
            log.flush()
            
            # Parse the input to retrieve question type, subject, and keyword.
            qtype, subject, keyword = parse_question(user_input)
            answer = ""
            # Log the parsed question components for debugging purposes.
            log.write(f"{counter} DEBUG: Parsed question - qtype: {qtype}, subject: {subject}, keyword: {keyword}\n")
            log.flush()
            
            if subject:
                # Log that the Wikipedia search is starting for the subject.
                log.write(f"{counter} DEBUG: Initiating Wikipedia search for subject: {subject}\n")
                log.flush()
                # Retrieve a summary from Wikipedia.
                summary = get_summary(subject)
                log.write(f"{counter} DEBUG: Raw Wikipedia summary: {summary}\n")
                log.flush()
                # Tokenize the subject to extract named entity information.
                ner_tags = token(subject)
                log.write(f"{counter} DEBUG: NER tags for subject: {ner_tags}\n")
                log.flush()
                # Determine if the subject refers to a person based on NER tags.
                is_person = any(tag in ['B-PERSON', 'I-PERSON'] for _, _, tag in ner_tags)
                
                # Process "when" questions.
                if qtype == "when":
                    # Check if the subject is a person or keyed to birthday/age.
                    if is_person or (keyword in ["born", "birthday", "age"]):
                        # Extract a date from the Wikipedia summary.
                        extracted = extract_date(summary)
                        log.write(f"{counter} DEBUG: Extracted date: {extracted}\n")
                        log.flush()
                        if extracted:
                            # Format the answer based on the extracted date.
                            answer = format_answer(qtype, subject, keyword, extracted)
                        else:
                            answer = "I am sorry, I don't know the answer."
                    else:
                        # For non-person subjects, use the summary directly.
                        answer = summary if summary else "I am sorry, I don't know the answer."
                
                # Process "where" questions.
                elif qtype == "where":
                    # Extract location details from the summary.
                    extracted = extract_location(summary, subject)
                    log.write(f"{counter} DEBUG: Extracted location: {extracted}\n")
                    log.flush()
                    if extracted:
                        answer = format_answer(qtype, subject, keyword, extracted)
                    else:
                        answer = "I am sorry, I don't know the answer."
                
                # Process "who" and "what" questions.
                elif qtype in ["who", "what"]:
                    answer = format_answer(qtype, subject, keyword, summary)
                else:
                    answer = "I am sorry, I don't know the answer."
            else:
                answer = "I am sorry, I don't know the answer."
            
            # Log and display the final answer.
            if answer == "":
                log.write(f"{counter} A) Answer not found.\n\n")
                log.flush()
                print("=> I am sorry, I don't know the answer.\n")
            else:
                log.write(f"{counter} A) {answer}\n\n")
                log.flush()
                print(f"=> {answer}\n")
            counter += 1
    except Exception as e:
        # Report any exceptions and log the error.
        print(f"Error: {e}")
        if log:
            log.write(f"ERROR: {e}\n")
            log.flush()
    finally:
        # Log the program ending time, flush, and close the log file.
        if log:
            log.write(f"--- Log ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log.flush()
            log.close()

if __name__ == "__main__":
    # Run the main function if this script is executed.
    main()