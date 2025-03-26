import wikipedia
import sys
import os
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import tree2conlltags

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('stopwords')

def main():
    log = None

    try:
        if len(sys.argv) < 2:
            sys.argv.append("/Users/vanshsetpal/Desktop/AIT 526/Programming Assignment 2/mylogfile.txt")

        log = open(sys.argv[1], "w+", encoding="utf-8")
        print("This is a QA system")
        print("It will try to answer questions that start with Who, What, When or Where.")
        print("Enter 'exit' to leave the program.\n")

        counter = 1
        while True:
            s = input("How can I help you today?\n").strip()
            if s.lower() == "exit":
                print("Thank you! Goodbye.")
                break

            log.write(f"{counter} Q) {s}\n")
            final = ""
            summary = ""
            flag = False

            if s:
                s_title = s.title()
                ner_tags = token(s_title)
                subject = question(s_title)

                if subject:
                    if 'B-PERSON' in [tag for _, _, tag in ner_tags] or 'I-PERSON' in [tag for _, _, tag in ner_tags]:
                        flag = True

                    summary = get_summary(subject)

                    if "When" in s_title and flag:
                        result = extract_date(summary)
                        if result:
                            if "Age" in s_title:
                                match = re.search(r'\d{4}', result)
                                if match:
                                    result = datetime.now().year - int(match.group())
                            final = format_answer(s_title, result)
                        else:
                            final = "I am sorry, I don't know the answer."
                    elif "Where" in s_title:
                        result = extract_location(summary, subject)
                        if result:
                            final = format_answer(s_title, result)
                        else:
                            final = "I am sorry, I don't know the answer."
                    else:
                        final = summary
                else:
                    final = "I am sorry, I don't know the answer."
            else:
                print("Please ask a valid question!")

            if final == "":
                log.write(f"{counter} A) Answer not found.\n\n")
                print("=> I am sorry, I don't know the answer.\n")
            else:
                try:
                    log.write(f"{counter} A) {final}\n\n")
                    print(f"=> {final}\n")
                except:
                    log.write(f"{counter} A) Answer not found.\n\n")
                    print("=> I am sorry, I don't know the answer.\n")

            counter += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if log:
            log.close()

def question(s):
    patterns = [
        [r'Where (Is|Was) (.+)', r'\2'],
        [r'Who (Is|Was) (.+)', r'\2'],
        [r'What (Is|Was) (.+) Age', r'\2'],
        [r'What (Is|Was) (.+)', r'\2'],
        [r'When (Is|Was) (.+) Born', r'\2'],
        [r'When (Is|Was) (.+) Birthday', r'\2']
    ]
    for pattern, group in patterns:
        match = re.match(pattern, s.rstrip('.!?'))
        if match:
            return match.group(2)
    return s  # fallback to full question

def token(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    tagged = pos_tag(filtered)
    tree = ne_chunk(tagged)
    return tree2conlltags(tree)

def get_summary(subject):
    try:
        return wikipedia.summary(subject, sentences=1)
    except:
        return ""

def extract_date(text):
    patterns = [
        r'\b\d{1,2} [A-Z][a-z]+ \d{4}\b',  # 22 February 1732
        r'\b[A-Z][a-z]+ \d{1,2}, \d{4}\b',  # February 22, 1732
        r'\b\d{4}\b'  # just year fallback
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

    for pattern in patterns:
        match = re.search(pattern, summary)
        if match:
            return match.group.strip()
    return ""

def format_answer(question_text, info):
    q = question_text.rstrip('.!?')
    q = re.sub(r'When (Is|Was) (.+) Born', r'\2 was born on ', q)
    q = re.sub(r'When (Is|Was) (.+) Birthday', r'\2\'s birthday is on ', q)
    q = re.sub(r'What (Is|Was) (.+) Age', r'\2\'s age is ', q)
    q = re.sub(r'Where (Is|Was) (.+)', r'\2 is located in ', q)
    return q + str(info) + '.'

main()
