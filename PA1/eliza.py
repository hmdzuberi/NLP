# AIT 526 Programming Assignment 1
# Hamaad Zuberi G01413525
# Aakiff
# Vansh


import re
import random

def eliza():
    # Initial greeting and asking for user's name
    print("[eliza] Hi, I'm a psychotherapist. What is your name?")
    name = input("=> [user] ")
    name = re.sub(r'^My name is ', '', name, flags=re.IGNORECASE).strip()  # Remove "My name is" from the input
    print(f"[eliza] Hi {name}. How can I help you today?")

    # Predefined responses for different patterns
    responses = {
        r'\b(I want|I wish)\b\s*(.*)': ["Why do you want {}?", "What would it mean if you got {}?"],
        r'\b(I am|I\'m)\s+(.*)': ["Why do you say you are {}?", "How long have you been {}?"],
        r'\b(I feel|I think)\s*(.*)': [f"{name}, tell me more about these feelings.", "Why do you think you feel this way?"],
        r'\b(love|hate)\b\s*(.*)': ["Tell me more about these feelings.", "How do you express these emotions?"],
        r'\b(because)\b\s*(.*)': [f"Is that the real reason, {name}?", "What other reasons might there be?"],
        r'\b(hello|hi|hey)\b\s*(.*)': [f"Hello {name}. How are you feeling today?", f"Hi {name}. What's on your mind?"],
        r'\b(computer|machine|program)\b\s*(.*)': [f"Are you really talking about me, {name}?", "Does it seem strange to talk to a computer?"],
        r'\b(mother|father|brother|sister|family)\b\s*(.*)': ["Tell me more about your family.", "How do you get along with your family?"],
        r'\b(depressed|sad|upset)\b\s*(.*)': ["I'm sorry to hear that. Can you tell me more about what's troubling you?"],
        r'\b(happy|excited|good)\b\s*(.*)': [f"That's great, {name}! What's making you feel so positive?"],
        r'\b(friend|friends)\b\s*(.*)': ["How do your friends make you feel?", "Do you have a best friend?"],
        r'\b(crave)\b\s*(.*)': ["Why don't you tell me more about your cravings."],
    }

    # Pronoun substitutions for responses
    pronouns = {
        'I': 'you',
        'me': 'you',
        'my': 'your',
        'mine': 'yours',
        'you': 'I',
        'your': 'my',
        'yours': 'mine',
        'am': 'are',
    }

    while True:
        user_input = input(f"=> [{name}] ")  # Get user input
        if user_input.lower() in ['bye', 'quit', 'exit']:  # Exit condition
            print("[eliza] It was nice talking to you. Take care!")
            break

        response = None
        for pattern, replies in responses.items():  # Check each pattern
            if re.search(pattern, user_input, re.IGNORECASE):
                response = random.choice(replies)  # Choose a random response
                match = re.search(pattern, user_input, re.IGNORECASE)
                if '{}' in response:  # If response requires formatting
                    phrase = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    phrase = ' '.join(pronouns.get(word, word) for word in phrase.split())  # Substitute pronouns
                    response = response.format(phrase)
                break

        if not response:  # Default responses if no pattern matches
            response = random.choice([
                "Can you elaborate on that?",
                "Interesting. How does that make you feel?",
                f"{name}, could you rephrase that?",
                "I'm not sure I understand. Can you explain differently?"
            ])

        print(f"[eliza] {response}")  # Print the response

if __name__ == "__main__":
    eliza()