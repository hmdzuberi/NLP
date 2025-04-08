#!/usr/bin/env python3
import re
import sys
import logging
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

def main():
    # Check if log file name is provided
    if len(sys.argv) != 2:
        print("Usage: python qa-system.py <logfile>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # Set up logging
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    
    # Welcome message
    print("*** This is a QA system by [YourName]. It will try to answer questions")
    print("that start with Who, What, When or Where. Enter \"exit\" to leave the")
    print("program.")
    
    # Main interaction loop
    while True:
        # Get user question
        user_input = input("=?> ")
        logging.info(f"User question: {user_input}")
        
        # Check for exit command
        if user_input.lower() == "exit":
            print("Thank you! Goodbye.")
            break
        
        # Process question
        answer = process_question(user_input)
        print(f"=> {answer}")
        logging.info(f"System answer: {answer}")

def process_question(question):
    # Identify question type (Who, What, When, Where)
    question_type = identify_question_type(question)
    logging.info(f"Question type: {question_type}")
    
    if question_type is None:
        return "I can only answer Who, What, When, or Where questions."
    
    # Extract the subject from the question
    subject = extract_subject(question, question_type)
    logging.info(f"Subject: {subject}")
    
    if not subject:
        return "I am sorry, I don't know the answer."
    
    # Generate answer patterns based on question type and subject
    answer_patterns = generate_answer_patterns(subject, question_type, question)
    logging.info(f"Answer patterns: {answer_patterns}")
    
    # Search Wikipedia for the subject
    try:
        # Search for the subject
        search_results = wikipedia.search(subject, results=5)
        logging.info(f"Search results: {search_results}")
        
        if not search_results:
            return "I am sorry, I don't know the answer."
        
        # Try each search result
        for result in search_results:
            try:
                page = wikipedia.page(result)
                content = page.content
                content = re.sub(r"\(.*?\)", "", content)
                logging.info(f"Examining Wikipedia page: {page.title}")
                logging.info(f"Page URL: {page.url}")
                
                # Log a sample of the content for debugging
                content_sample = content[:500] + "..." if len(content) > 500 else content
                logging.info(f"Content sample: {content_sample}")
                
                # Look for answer patterns in the content
                answer = find_answer(content, answer_patterns, question_type, subject, question)
                
                if answer:
                    return answer
            
            except DisambiguationError as e:
                logging.info(f"Disambiguation for {result}: {str(e)}")
                # Try the first option
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0])
                        content = page.content
                        logging.info(f"Trying disambiguation option: {e.options[0]}")
                        
                        answer = find_answer(content, answer_patterns, question_type, subject, question)
                        
                        if answer:
                            return answer
                    except Exception as inner_e:
                        logging.error(f"Error with disambiguation option: {str(inner_e)}")
            
            except PageError as e:
                logging.info(f"Page error for {result}: {str(e)}")
                continue
            
            except Exception as e:
                logging.error(f"Unexpected error for {result}: {str(e)}")
                continue
        
        return "I am sorry, I don't know the answer."
    
    except Exception as e:
        logging.error(f"Error searching Wikipedia: {str(e)}")
        return "I am sorry, I don't know the answer."

def identify_question_type(question):
    question = question.strip().lower()
    
    if question.startswith("who"):
        return "who"
    elif question.startswith("what"):
        return "what"
    elif question.startswith("when"):
        return "when"
    elif question.startswith("where"):
        return "where"
    else:
        return None

def extract_subject(question, question_type):
    # Remove the question word
    question = question.strip()
    question = re.sub(r'^' + question_type + r'\s+', '', question, flags=re.IGNORECASE)
    
    # Remove question mark
    question = question.rstrip('?')
    
    # Remove common auxiliary verbs at the beginning
    question = re.sub(r'^\s*(is|are|was|were|do|does|did)\s+', '', question, flags=re.IGNORECASE)
    
    # Special case for "When was X born"
    if question_type == "when" and re.search(r'\bborn\b', question, flags=re.IGNORECASE):
        question = re.sub(r'\s+born\s*$', '', question, flags=re.IGNORECASE)
    
    return question.strip()

def generate_answer_patterns(subject, question_type, original_question):
    patterns = []
    
    if question_type == "who":
        patterns = [
            f"{subject} is",
            f"{subject} was",
            f"{subject} is a",
            f"{subject} was a",
            f"{subject} is an",
            f"{subject} was an",
            f"{subject}, is",
            f"{subject}, was",
            f"{subject} has been",
            f"{subject} became",
        ]
    
    elif question_type == "what":
        patterns = [
            f"{subject} is",
            f"{subject} are",
            f"{subject} was",
            f"{subject} were",
            f"{subject} refers to",
            f"{subject} is defined as",
            f"A {subject} is",
            f"An {subject} is",
            f"The {subject} is",
            f"{subject} means",
            f"{subject} consists of",
            f"{subject} comprises",
        ]
    
    elif question_type == "when":
        if re.search(r'\bborn\b', original_question, flags=re.IGNORECASE):
            patterns = [
                f"{subject} was born on",
                f"{subject} was born in",
                f"{subject}, born on",
                f"{subject}, born in",
                f"{subject} (born",
            ]
        elif re.search(r'\bdie(d)?\b', original_question, flags=re.IGNORECASE):
            patterns = [
                f"{subject} died on",
                f"{subject} died in",
                f"{subject}, died on",
                f"{subject}, died in",
                f"{subject} (died",
            ]
        else:
            patterns = [
                f"{subject} occurred on",
                f"{subject} occurred in",
                f"{subject} happened on",
                f"{subject} happened in",
                f"{subject} took place on",
                f"{subject} took place in",
                f"{subject} is on",
                f"{subject} is in",
                f"{subject} was on",
                f"{subject} was in",
                f"{subject} is celebrated on",
                f"{subject} is observed on",
                f"{subject} began on",
                f"{subject} began in",
            ]
    
    elif question_type == "where":
        patterns = [
            f"{subject} is located in",
            f"{subject} is located at",
            f"{subject} is situated in",
            f"{subject} is situated at",
            f"{subject} is found in",
            f"{subject} is found at",
            f"{subject} is in",
            f"{subject} is at",
            f"{subject}, located in",
            f"{subject}, located at",
            f"{subject}'s location is",
            f"{subject} can be found in",
            f"{subject} can be found at",
            f"{subject} is based in",
            f"{subject} is based at",
        ]
    
    return patterns

def find_answer(content, answer_patterns, question_type, subject, original_question):
    logging.info(f"Searching for patterns in content")
    
    # Try each pattern
    for pattern in answer_patterns:
        # Escape pattern for regex
        escaped_pattern = re.escape(pattern)
        # Find all occurrences of pattern
        pattern_regex = escaped_pattern + r'([^\.]*)\.'
        matches = re.finditer(pattern_regex, content, re.IGNORECASE)
        
        for match in matches:
            print(f"Match found: {match.group(0)}")
        
        for match in matches:
            answer_text = match.group(1).strip()
            logging.info(f"Pattern '{pattern}' matched: {answer_text}")
            
            # Clean up the answer text
            answer_text = re.sub(r'\[\d+\]', '', answer_text)  # Remove citations like [1]
            answer_text = re.sub(r'\s+', ' ', answer_text).strip()  # Normalize whitespace
            
            # Format the answer as a complete sentence
            if answer_text:
                formatted_answer = format_answer(pattern, answer_text, question_type, subject, original_question)
                if formatted_answer:
                    return formatted_answer
    
    return None

def format_answer(pattern, answer_text, question_type, subject, original_question):
    # Remove any additional information in parentheses
    answer_text = re.sub(r'\([^)]*\)', '', answer_text)
    answer_text = answer_text.strip()
    
    if not answer_text:
        return None
    
    # Format based on question type
    if question_type == "who":
        return f"{subject} is {answer_text}."
    
    elif question_type == "what":
        if pattern.startswith(("A ", "An ", "The ")):
            article = pattern.split()[0]
            return f"{article} {subject} is {answer_text}."
        else:
            return f"{subject} is {answer_text}."
    
    elif question_type == "when":
        if re.search(r'\bborn\b', original_question, flags=re.IGNORECASE):
            return f"{subject} was born {answer_text}."
        elif re.search(r'\bdie(d)?\b', original_question, flags=re.IGNORECASE):
            return f"{subject} died {answer_text}."
        else:
            return f"{subject} occurred {answer_text}."
    
    elif question_type == "where":
        return f"{subject} is located {answer_text}."
    
    return None

if __name__ == "__main__":
    main()