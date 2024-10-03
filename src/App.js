import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [image, setImage] = useState(null);
  const [messages, setMessages] = useState([]);
  const [language, setLanguage] = useState('en-US'); // Default language set to English
  const [recognition, setRecognition] = useState(null);
  const [sensitivity, setSensitivity] = useState(1); // Slider value for mic sensitivity

  // Function to handle language change
  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

  // Function to handle image upload
  const handleImageUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      const uploadedImage = URL.createObjectURL(e.target.files[0]);
      setImage(uploadedImage);

      // Add image message to the chat history
      setMessages((prevMessages) => [
        ...prevMessages,
        { type: 'image', content: uploadedImage },
      ]);
    }
  };

  // Function to handle chat message send
  const handleSendMessage = () => {
    if (inputText.trim() !== '') {
      setMessages((prevMessages) => [
        ...prevMessages,
        { type: 'text', content: inputText },
      ]);
      speakText(inputText); // Speak out the message in the selected language
      setInputText(''); // Clear the input field
    }
  };

  // Function to speak text using Web Speech API
  const speakText = (text) => {
    const speech = new SpeechSynthesisUtterance(text);
    speech.lang = language; // Set the language to selected value
    speech.pitch = 1;
    speech.rate = 1;
    speech.volume = sensitivity; // Set volume based on sensitivity slider
    window.speechSynthesis.speak(speech);
  };

  // Function to start speech recognition
  const startRecognition = () => {
    if (recognition) {
      recognition.start();
    }
  };

  // Function to handle speech recognition results
  const handleRecognitionResult = (event) => {
    const transcript = event.results[0][0].transcript;
    setInputText(transcript);
    handleSendMessage();
  };

  // Setup speech recognition when component mounts
  useEffect(() => {
    // Check for SpeechRecognition API support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.lang = language; // Set recognition language to selected value
      recognitionInstance.interimResults = false;
      recognitionInstance.onresult = handleRecognitionResult;
      setRecognition(recognitionInstance);
    } else {
      console.error('Speech Recognition API is not supported in this browser.');
      alert('Speech Recognition API is not supported in this browser. Please use Google Chrome.');
    }

    // Speak welcome message in the selected language when component mounts
    speakText('Welcome to the chatbot! How can I assist you today?');
  }, [language, sensitivity]);

  return (
    <div className="App">
      <div className="container">
        {/* Sidebar for Chat History */}
        <div className="sidebar">
          <h2>Chat History</h2>
          <div className="chat-history">
            {messages.map((message, index) => (
              <div key={index} className="history-item">
                {message.type === 'text' ? (
                  <p>{message.content}</p>
                ) : (
                  <img src={message.content} alt="Uploaded" className="history-image" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Main Chat Section */}
        <div className="main-chat">
          <h1>Chatbot Interface</h1>

          {/* Language Selection */}
          <div className="language-selection">
            <label htmlFor="language">Select Language: </label>
            <select id="language" value={language} onChange={handleLanguageChange}>
              <option value="en-US">English</option>
              <option value="es-ES">Español</option>
              <option value="fr-FR">Français</option>
              <option value="hi-IN">हिन्दी</option>
            </select>
          </div>

          {/* Chat Box */}
          <div className="chat-box">
            <div className="chat-history">
              {messages.map((message, index) => (
                <div key={index} className={`message ${message.type}`}>
                  {message.type === 'text' ? (
                    <p>{message.content}</p>
                  ) : (
                    <img src={message.content} alt="Uploaded" className="uploaded-image" />
                  )}
                </div>
              ))}
            </div>

            {/* Chat Input and Send Button */}
            <div className="chat-input-section">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type your message here..."
                className="chat-input"
              />
              <button onClick={handleSendMessage} className="send-btn">
                Send
              </button>
            </div>

            {/* Voice Input Button */}
            <div className="voice-input-section">
              <button onClick={startRecognition} className="voice-btn">🎤 Speak</button>
            </div>

            {/* Image Upload Section */}
            <div className="image-upload-section">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="file-upload"
              />
            </div>

            {/* Microphone Sensitivity Slider */}
            <div className="mic-sensitivity">
              <label htmlFor="sensitivity">Mic Sensitivity:</label>
              <input
                type="range"
                id="sensitivity"
                min="0"
                max="1"
                step="0.1"
                value={sensitivity}
                onChange={(e) => setSensitivity(e.target.value)}
                className="slider"
              />
              <span>{sensitivity}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
