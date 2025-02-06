import re
import random

def eliza():
    print("[eliza] Hi, I'm a psychotherapist. What is your name?")
    name = input("=> [user] ")
    name = re.sub(r'^My name is ', '', name, flags=re.IGNORECASE).strip()
    print(f"[eliza] Hi {name}. How can I help you today?")

    responses = {
        r'\b(I want|I wish)\b\s*(.*)': ["Why do you want {}?", "What would it mean if you got {}?"],
        r'\b(I am|I\'m)\s+(.*)': ["Why do you say you are {}?", "How long have you been {}?"],
        r'\b(I feel|I think)\s*(.*)': ["Tell me more about these feelings.", "Why do you think you feel this way?"],
        r'\b(love|hate)\b\s*(.*)': ["Tell me more about these feelings.", "How do you express these emotions?"],
        r'\b(because)\b\s*(.*)': ["Is that the real reason?", "What other reasons might there be?"],
        r'\b(hello|hi|hey)\b\s*(.*)': ["Hello {}. How are you feeling today?", "Hi {}. What's on your mind?"],
        r'\b(computer|machine|program)\b\s*(.*)': ["Are you really talking about me?", "Does it seem strange to talk to a computer?"],
        r'\b(mother|father|brother|sister)\b\s*(.*)': ["Tell me more about your family.", "How do you get along with your family?"],
        r'\b(depressed|sad|upset)\b\s*(.*)': ["I'm sorry to hear that. Can you tell me more about what's troubling you?"],
        r'\b(happy|excited|good)\b\s*(.*)': ["That's great! What's making you feel so positive?"],
        r'\b(friend|friends)\b\s*(.*)': ["How do your friends make you feel?", "Do you have a best friend?"],
        r'\b(crave|want|need)\b\s*(.*)': ["What would happen if you got {}?", "Why do you want {}?"],
    }

    pronouns = {
        'I': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours',
        'you': 'I', 'your': 'my', 'yours': 'mine'
    }

    while True:
        user_input = input(f"=> [{name}] ")
        if user_input.lower() == 'quit':
            print("[eliza] It was nice talking to you. Take care!")
            break

        response = None
        for pattern, replies in responses.items():
            if re.search(pattern, user_input, re.IGNORECASE):
                response = random.choice(replies)
                match = re.search(pattern, user_input, re.IGNORECASE)
                if '{}' in response:
                    phrase = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    phrase = ' '.join(pronouns.get(word, word) for word in phrase.split())
                    response = response.format(phrase)
                break

        if not response:
            response = random.choice([
                "Can you elaborate on that?",
                "I see. Tell me more.",
                "Interesting. How does that make you feel?",
                f"{name}, could you rephrase that?",
                "I'm not sure I understand. Can you explain differently?"
            ])

        print(f"[eliza] {response}")

if __name__ == "__main__":
    eliza()