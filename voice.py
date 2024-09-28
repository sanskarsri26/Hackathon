import speech_recognition as sr

def main():
    # Create a Recognizer instance
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise. Please wait...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        print("Listening... Please say something:")
        audio = recognizer.listen(source)  # Capture audio

        try:
            # Recognize speech using Google Web Speech API (offline options can be explored)
            text = recognizer.recognize_google(audio)
            print("You said: " + text)  # Print the recognized text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    main()
