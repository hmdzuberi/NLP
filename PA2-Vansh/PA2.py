import wikipedia
import sys
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import tree2conlltags

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

def parse_question(question):
    q = question.strip().rstrip('.!?')
    pattern = re.compile(
        r'^(Who|What|When|Where)\s+(?:is|was)\s+(.+?)(?:\s+(born|birthday|age))?$',
        re.IGNORECASE
    )
    match = pattern.match(q)
    if match:
        qtype = match.group(1).lower()  # 'who', 'what', 'when', 'where'
        subject = match.group(2).strip()
        keyword = match.group(3).lower() if match.group(3) else None
        return qtype, subject, keyword
    return None, question, None

def token(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged)
    return tree2conlltags(tree)

def get_summary(subject):
    try:
        return wikipedia.summary(subject, sentences=1)
    except Exception:
        return ""

def extract_date(text):
    patterns = [
        r'\b\d{1,2} [A-Z][a-z]+ \d{4}\b',   # e.g., 22 February 1732
        r'\b[A-Z][a-z]+ \d{1,2}, \d{4}\b',    # e.g., February 22, 1732
        r'\b\d{4}\b'                         # fallback: year only
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if "to" not in m and "-" not in m:
                return m
    return ""

def extract_location(summary, subject):
    summary = summary.lower()
    subject_lower = subject.lower()

    patterns = [
        rf"{subject_lower}.*?was born in ([a-zA-Z ,\-]+)",
        rf"{subject_lower}.*?is located in ([a-zA-Z ,\-]+)",
        rf"{subject_lower}.*?is situated in ([a-zA-Z ,\-]+)",
        rf"{subject_lower}.*?is in ([a-zA-Z ,\-]+)"
    ]

    # More general regex to capture locations after prepositions
    prepositions = r"(in|at|near|on|from|by|to)"
    location_pattern = rf"\b{prepositions}\s+((?:[A-Za-z0-9\s,\.\-\'])+)"

    match = re.search(location_pattern, summary)
    if match:
        return match.group(2).strip()

    return ""

def format_answer(qtype, subject, keyword, info):
    if not info:
        return "I am sorry, I don't know the answer."
    if qtype == "when":
        if keyword == "age":
            year_match = re.search(r'\d{4}', info)
            if year_match:
                age = datetime.now().year - int(year_match.group())
                return f"{subject}'s age is {age}."
            else:
                return "I am sorry, I don't know the answer."
        elif keyword in ["born", "birthday"]:
            return f"{subject} was born on {info}."
        else:
            return f"{subject} is associated with {info}."
    elif qtype == "where":
        return f"{subject} is located in {info}."
    elif qtype in ["who", "what"]:
        return info if info.endswith('.') else info + '.'
    else:
        return info

def main():
    log = None
    try:
        if len(sys.argv) < 2:
            sys.argv.append("mylogfile.txt")
        log = open(sys.argv[1], "a+", encoding="utf-8")
        log.write(f"--- Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
        log.flush()
        print("This is a QA system")how 
        print("It will try to answer questions that start with Who, What, When or Where.")
        print("Enter 'exit' to leave the program.\n")
        counter = 1
        while True:
            user_input = input("Shoot me a question?\n").strip()
            if user_input.lower() == "exit":
                print("Thank you! Goodbye.")
                log.write("User exited the program.\n")
                log.flush()
                break
            log.write(f"{counter} Q) {user_input}\n")
            log.flush()
            qtype, subject, keyword = parse_question(user_input)
            answer = ""
            # Log the parsed question details for debugging.
            log.write(f"{counter} DEBUG: Parsed question - qtype: {qtype}, subject: {subject}, keyword: {keyword}\n")
            log.flush()
            if subject:
                log.write(f"{counter} DEBUG: Initiating Wikipedia search for subject: {subject}\n")
                log.flush()
                summary = get_summary(subject)
                log.write(f"{counter} DEBUG: Raw Wikipedia summary: {summary}\n")
                log.flush()
                ner_tags = token(subject)
                log.write(f"{counter} DEBUG: NER tags for subject: {ner_tags}\n")
                log.flush()
                is_person = any(tag in ['B-PERSON', 'I-PERSON'] for _, _, tag in ner_tags)
                if qtype == "when":
                    if is_person or (keyword in ["born", "birthday", "age"]):
                        extracted = extract_date(summary)
                        log.write(f"{counter} DEBUG: Extracted date: {extracted}\n")
                        log.flush()
                        if extracted:
                            answer = format_answer(qtype, subject, keyword, extracted)
                        else:
                            answer = "I am sorry, I don't know the answer."
                    else:
                        answer = summary if summary else "I am sorry, I don't know the answer."
                elif qtype == "where":
                    extracted = extract_location(summary, subject)
                    log.write(f"{counter} DEBUG: Extracted location: {extracted}\n")
                    log.flush()
                    if extracted:
                        answer = format_answer(qtype, subject, keyword, extracted)
                    else:
                        answer = "I am sorry, I don't know the answer."
                elif qtype in ["who", "what"]:
                    answer = format_answer(qtype, subject, keyword, summary)
                else:
                    answer = "I am sorry, I don't know the answer."
            else:
                answer = "I am sorry, I don't know the answer."
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
        print(f"Error: {e}")
        if log:
            log.write(f"ERROR: {e}\n")
            log.flush()
    finally:
        if log:
            log.write(f"--- Log ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log.flush()
            log.close()

if __name__ == "__main__":
    main()