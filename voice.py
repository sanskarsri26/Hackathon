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
            # Recognize speech using Google Web Speech API
            recognized_text = recognizer.recognize_google(audio)
            print("You said: " + recognized_text)  # Print the recognized text
            
            # Now you can use recognized_text variable for further processing
            return recognized_text  # Optionally return the recognized text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None  # Return None if the speech could not be understood
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None  # Return None on request error

if __name__ == "__main__":
    result = main()  # Store the result in a variable
    if result:
        # You can now use the recognized speech stored in the 'result' variable
        print(f"Stored recognized text: {result}")