from pytector import PromptInjectionDetector


detector = PromptInjectionDetector(model_name_or_url="deberta")

is_injection, probability = detector.detect_injection("Your suspicious prompt here")
print(f"Is injection: {is_injection}, Probability: {probability}")
