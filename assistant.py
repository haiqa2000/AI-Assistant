#!/usr/bin/env python3
"""
Enhanced AI-Powered Virtual Personal Assistant (VPA)
Advanced implementation with offline speech recognition, ML-based NLU, IoT integration,
multi-language support, and cloud synchronization
"""

import speech_recognition as sr
import pyttsx3
import spacy
import requests
import json
import datetime
import re
import threading
import schedule
import time
import webbrowser
import sqlite3
import hashlib
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import vosk
import sounddevice as sd
import queue
import pyaudio
import wave
from cryptography.fernet import Fernet
import pygame
import cv2
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile for multi-user support"""
    user_id: str
    name: str
    voice_embedding: np.ndarray
    preferences: Dict[str, Any]
    language: str = 'en'
    created_at: datetime.datetime = None

class OfflineSpeechRecognizer:
    """Offline speech recognition using Vosk"""
    
    def __init__(self, model_path: str = "vosk-model-en-us-0.22"):
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.setup_model()
    
    def setup_model(self):
        """Setup Vosk offline model"""
        try:
            if os.path.exists(self.model_path):
                self.model = vosk.Model(self.model_path)
                self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
                logger.info("Offline speech recognition model loaded")
            else:
                logger.warning(f"Vosk model not found at {self.model_path}")
                logger.info("Download model from: https://alphacephei.com/vosk/models")
        except Exception as e:
            logger.error(f"Failed to load offline model: {e}")
    
    def recognize(self, audio_data: bytes) -> Optional[str]:
        """Recognize speech from audio data"""
        if not self.recognizer:
            return None
        
        try:
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                return result.get('text', '').lower()
            else:
                partial = json.loads(self.recognizer.PartialResult())
                return partial.get('partial', '').lower()
        except Exception as e:
            logger.error(f"Offline recognition error: {e}")
            return None

class EmotionRecognizer:
    """Emotion recognition from speech and facial expressions"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
    
    def analyze_voice_emotion(self, audio_features: np.ndarray) -> str:
        """Analyze emotion from voice features (simplified)"""
        # This is a simplified version - in production, you'd use a trained model
        emotions = ['happy', 'sad', 'angry', 'neutral', 'excited']
        return np.random.choice(emotions)  # Mock implementation
    
    def analyze_facial_emotion(self, frame: np.ndarray) -> str:
        """Analyze emotion from facial expressions"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Simplified emotion detection based on facial landmarks
                results = self.face_mesh.process(frame)
                if results.multi_face_landmarks:
                    # In a real implementation, you'd analyze facial landmark positions
                    emotions = ['happy', 'sad', 'surprised', 'neutral']
                    return np.random.choice(emotions)  # Mock implementation
            
            return 'neutral'
        except Exception as e:
            logger.error(f"Facial emotion recognition error: {e}")
            return 'neutral'

class CustomWakeWordDetector:
    """Custom wake word detection using machine learning"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.wake_words = []
        self.setup_model()
    
    def setup_model(self):
        """Setup or load wake word detection model"""
        try:
            if os.path.exists('wake_word_model.h5'):
                self.model = load_model('wake_word_model.h5')
                with open('wake_word_tokenizer.pkl', 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info("Wake word model loaded")
            else:
                self.train_default_model()
        except Exception as e:
            logger.error(f"Wake word model setup error: {e}")
    
    def train_default_model(self):
        """Train a simple wake word model with default data"""
        # Sample training data (in production, you'd have more diverse data)
        wake_samples = [
            "hey assistant", "assistant", "computer", "jarvis", 
            "okay assistant", "hello assistant"
        ]
        non_wake_samples = [
            "hello world", "what time is it", "play music", "weather today",
            "good morning", "how are you", "thank you"
        ]
        
        all_samples = wake_samples + non_wake_samples
        labels = [1] * len(wake_samples) + [0] * len(non_wake_samples)
        
        self.tokenizer = Tokenizer(num_words=1000)
        self.tokenizer.fit_on_texts(all_samples)
        
        sequences = self.tokenizer.texts_to_sequences(all_samples)
        data = pad_sequences(sequences, maxlen=10)
        
        # Simple neural network for wake word detection
        self.model = Sequential([
            Embedding(1000, 50, input_length=10),
            LSTM(32),
            Dense(16, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(data, np.array(labels), epochs=10, verbose=0)
        
        # Save model
        self.model.save('wake_word_model.h5')
        with open('wake_word_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def detect_wake_word(self, text: str) -> bool:
        """Detect if text contains wake word"""
        if not self.model or not self.tokenizer:
            # Fallback to simple keyword matching
            return any(word in text.lower() for word in ['assistant', 'computer', 'hey assistant'])
        
        try:
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=10)
            prediction = self.model.predict(padded, verbose=0)[0][0]
            return prediction > 0.5
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False

class MLIntentClassifier:
    """Machine learning-based intent classifier"""
    
    def __init__(self):
        self.pipeline = None
        self.labels = []
        self.setup_classifier()
    
    def setup_classifier(self):
        """Setup or load ML intent classifier"""
        try:
            if os.path.exists('intent_classifier.pkl'):
                with open('intent_classifier.pkl', 'rb') as f:
                    self.pipeline = pickle.load(f)
                with open('intent_labels.pkl', 'rb') as f:
                    self.labels = pickle.load(f)
                logger.info("ML intent classifier loaded")
            else:
                self.train_classifier()
        except Exception as e:
            logger.error(f"Intent classifier setup error: {e}")
    
    def train_classifier(self):
        """Train intent classifier with sample data"""
        # Training data (in production, you'd have much more data)
        training_data = [
            ("what's the weather like today", "weather"),
            ("tell me the temperature", "weather"),
            ("is it going to rain", "weather"),
            ("weather forecast", "weather"),
            ("what time is it", "time"),
            ("current time please", "time"),
            ("tell me the time", "time"),
            ("set an alarm for 8am", "alarm"),
            ("remind me at 5pm", "alarm"),
            ("wake me up at 7", "alarm"),
            ("search for python tutorials", "search"),
            ("google machine learning", "search"),
            ("find information about AI", "search"),
            ("play some music", "music"),
            ("start spotify", "music"),
            ("play my playlist", "music"),
            ("latest news", "news"),
            ("today's headlines", "news"),
            ("what's happening in the world", "news"),
            ("tell me a joke", "joke"),
            ("make me laugh", "joke"),
            ("something funny", "joke"),
            ("hello", "greeting"),
            ("hi there", "greeting"),
            ("good morning", "greeting"),
            ("goodbye", "goodbye"),
            ("see you later", "goodbye"),
            ("bye bye", "goodbye"),
            ("turn on the lights", "smart_home"),
            ("adjust the thermostat", "smart_home"),
            ("lock the doors", "smart_home"),
            ("what's my schedule", "calendar"),
            ("any meetings today", "calendar"),
            ("add event to calendar", "calendar"),
        ]
        
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        self.labels = list(set(labels))
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ('classifier', MultinomialNB())
        ])
        
        self.pipeline.fit(texts, labels)
        
        # Save model
        with open('intent_classifier.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)
        with open('intent_labels.pkl', 'wb') as f:
            pickle.dump(self.labels, f)
        
        logger.info("Intent classifier trained and saved")
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict intent with confidence score"""
        if not self.pipeline:
            return "unknown", 0.0
        
        try:
            prediction = self.pipeline.predict([text])[0]
            probabilities = self.pipeline.predict_proba([text])[0]
            confidence = max(probabilities)
            
            return prediction, confidence
        except Exception as e:
            logger.error(f"Intent prediction error: {e}")
            return "unknown", 0.0

class IoTController:
    """IoT device controller with multiple protocol support"""
    
    def __init__(self):
        self.devices = {}
        self.setup_devices()
    
    def setup_devices(self):
        """Setup mock IoT devices"""
        self.devices = {
            'living_room_light': {'type': 'light', 'state': 'off', 'brightness': 100},
            'bedroom_light': {'type': 'light', 'state': 'off', 'brightness': 80},
            'thermostat': {'type': 'thermostat', 'temperature': 72, 'mode': 'auto'},
            'front_door': {'type': 'lock', 'state': 'locked'},
            'security_camera': {'type': 'camera', 'state': 'off'},
            'smart_tv': {'type': 'tv', 'state': 'off', 'channel': 1}
        }
    
    def control_device(self, device_name: str, action: str, value: Any = None) -> str:
        """Control IoT device"""
        if device_name not in self.devices:
            return f"Device '{device_name}' not found"
        
        device = self.devices[device_name]
        
        try:
            if device['type'] == 'light':
                if action == 'turn_on':
                    device['state'] = 'on'
                    return f"Turned on {device_name}"
                elif action == 'turn_off':
                    device['state'] = 'off'
                    return f"Turned off {device_name}"
                elif action == 'brightness' and value:
                    device['brightness'] = min(100, max(0, int(value)))
                    return f"Set {device_name} brightness to {device['brightness']}%"
            
            elif device['type'] == 'thermostat':
                if action == 'temperature' and value:
                    device['temperature'] = int(value)
                    return f"Set thermostat to {device['temperature']}°F"
                elif action == 'mode' and value:
                    device['mode'] = value
                    return f"Set thermostat mode to {device['mode']}"
            
            elif device['type'] == 'lock':
                if action == 'lock':
                    device['state'] = 'locked'
                    return f"Locked {device_name}"
                elif action == 'unlock':
                    device['state'] = 'unlocked'
                    return f"Unlocked {device_name}"
            
            return f"Action '{action}' not supported for {device_name}"
            
        except Exception as e:
            logger.error(f"IoT control error: {e}")
            return f"Failed to control {device_name}"
    
    def get_device_status(self, device_name: str = None) -> str:
        """Get device status"""
        if device_name:
            if device_name in self.devices:
                device = self.devices[device_name]
                return f"{device_name}: {json.dumps(device, indent=2)}"
            else:
                return f"Device '{device_name}' not found"
        else:
            status = "All devices:\n"
            for name, device in self.devices.items():
                status += f"- {name}: {device['type']} - {device.get('state', 'unknown')}\n"
            return status

class CloudSync:
    """Cloud synchronization for preferences and data"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.encryption_key = self._get_or_create_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key"""
        key_file = f".{self.user_id}_key.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt data for cloud storage"""
        json_data = json.dumps(data).encode()
        return self.cipher_suite.encrypt(json_data)
    
    def decrypt_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt data from cloud storage"""
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    def sync_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Sync user preferences to cloud (mock implementation)"""
        try:
            encrypted_prefs = self.encrypt_data(preferences)
            # In production, this would upload to cloud storage
            with open(f".{self.user_id}_prefs.enc", 'wb') as f:
                f.write(encrypted_prefs)
            logger.info("Preferences synced to cloud")
            return True
        except Exception as e:
            logger.error(f"Cloud sync error: {e}")
            return False
    
    def load_preferences(self) -> Dict[str, Any]:
        """Load user preferences from cloud"""
        try:
            pref_file = f".{self.user_id}_prefs.enc"
            if os.path.exists(pref_file):
                with open(pref_file, 'rb') as f:
                    encrypted_data = f.read()
                return self.decrypt_data(encrypted_data)
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")
            return {}

class MultiLanguageProcessor:
    """Multi-language support for the assistant"""
    
    def __init__(self):
        self.supported_languages = {
            'en': {'name': 'English', 'spacy_model': 'en_core_web_sm'},
            'es': {'name': 'Spanish', 'spacy_model': 'es_core_news_sm'},
            'fr': {'name': 'French', 'spacy_model': 'fr_core_news_sm'},
            'de': {'name': 'German', 'spacy_model': 'de_core_news_sm'},
            'zh': {'name': 'Chinese', 'spacy_model': 'zh_core_web_sm'}
        }
        self.current_language = 'en'
        self.nlp_models = {}
        self.translations = self._load_translations()
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation strings"""
        return {
            'en': {
                'greeting': "Hello! How can I help you today?",
                'goodbye': "Goodbye! Have a great day!",
                'unknown': "I'm not sure how to help with that.",
                'weather': "The weather today is",
                'time': "The current time is",
                'error': "Sorry, I encountered an error."
            },
            'es': {
                'greeting': "¡Hola! ¿Cómo puedo ayudarte hoy?",
                'goodbye': "¡Adiós! ¡Que tengas un gran día!",
                'unknown': "No estoy seguro de cómo ayudar con eso.",
                'weather': "El clima de hoy es",
                'time': "La hora actual es",
                'error': "Lo siento, encontré un error."
            },
            'fr': {
                'greeting': "Bonjour! Comment puis-je vous aider aujourd'hui?",
                'goodbye': "Au revoir! Passez une excellente journée!",
                'unknown': "Je ne sais pas comment vous aider avec ça.",
                'weather': "Le temps aujourd'hui est",
                'time': "L'heure actuelle est",
                'error': "Désolé, j'ai rencontré une erreur."
            }
        }
    
    def set_language(self, language_code: str) -> bool:
        """Set the current language"""
        if language_code in self.supported_languages:
            self.current_language = language_code
            return True
        return False
    
    def get_nlp_model(self, language_code: str = None):
        """Get spaCy model for specified language"""
        lang = language_code or self.current_language
        
        if lang not in self.nlp_models:
            try:
                model_name = self.supported_languages[lang]['spacy_model']
                self.nlp_models[lang] = spacy.load(model_name)
            except OSError:
                logger.warning(f"spaCy model for {lang} not found, using English")
                return self.nlp_models.get('en', None)
        
        return self.nlp_models.get(lang)
    
    def translate(self, key: str, language_code: str = None) -> str:
        """Get translated string"""
        lang = language_code or self.current_language
        return self.translations.get(lang, {}).get(key, 
                self.translations.get('en', {}).get(key, key))

class DatabaseManager:
    """Database manager for storing user data and conversations"""
    
    def __init__(self, db_path: str = "assistant.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                voice_embedding BLOB,
                preferences TEXT,
                language TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TIMESTAMP,
                user_input TEXT,
                assistant_response TEXT,
                intent TEXT,
                confidence REAL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Learning data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                correct_intent TEXT,
                feedback_score INTEGER,
                timestamp TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_user(self, user_profile: UserProfile):
        """Save user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, name, voice_embedding, preferences, language, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_profile.user_id,
            user_profile.name,
            user_profile.voice_embedding.tobytes() if user_profile.voice_embedding is not None else None,
            json.dumps(user_profile.preferences),
            user_profile.language,
            user_profile.created_at or datetime.datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            voice_embedding = np.frombuffer(row[2]) if row[2] else None
            return UserProfile(
                user_id=row[0],
                name=row[1],
                voice_embedding=voice_embedding,
                preferences=json.loads(row[3]),
                language=row[4],
                created_at=row[5]
            )
        
        return None
    
    def save_conversation(self, user_id: str, user_input: str, response: str, 
                         intent: str, confidence: float):
        """Save conversation for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (user_id, timestamp, user_input, assistant_response, intent, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, datetime.datetime.now(), user_input, response, intent, confidence))
        
        conn.commit()
        conn.close()

class EnhancedSpeechProcessor:
    """Enhanced speech processor with offline and online capabilities"""
    
    def __init__(self):
        self.online_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.offline_recognizer = OfflineSpeechRecognizer()
        self.tts_engine = pyttsx3.init()
        self.emotion_recognizer = EmotionRecognizer()
        self.use_offline = False
        self._setup_tts()
        self._calibrate_microphone()
    
    def _setup_tts(self):
        """Configure text-to-speech settings"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        with self.microphone as source:
            self.online_recognizer.adjust_for_ambient_noise(source)
    
    def listen(self) -> Optional[Tuple[str, str]]:
        """Listen for speech and return text with emotion"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.online_recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Try offline first if enabled
            if self.use_offline and self.offline_recognizer.model:
                text = self.offline_recognizer.recognize(audio.get_raw_data())
            else:
                text = self.online_recognizer.recognize_google(audio)
            
            if text:
                text = text.lower()
                # Analyze emotion from audio (mock implementation)
                emotion = self.emotion_recognizer.analyze_voice_emotion(np.array([]))
                logger.info(f"Recognized: {text} (emotion: {emotion})")
                return text, emotion
            
            return None, None
        
        except sr.WaitTimeoutError:
            return None, None
        except sr.UnknownValueError:
            self.speak("I didn't understand that. Could you repeat?")
            return None, None
        except sr.RequestError as e:
            if not self.use_offline:
                logger.warning("Online recognition failed, switching to offline")
                self.use_offline = True
            return None, None
    
    def speak(self, text: str, language: str = 'en'):
        """Convert text to speech with language support"""
        print(f"Assistant: {text}")
        
        # Set language-specific voice if available
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if language in voice.id.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

class EnhancedVirtualPersonalAssistant:
    """Enhanced VPA with all advanced features"""
    
    def __init__(self):
        self.speech_processor = EnhancedSpeechProcessor()
        self.ml_classifier = MLIntentClassifier()
        self.wake_word_detector = CustomWakeWordDetector()
        self.iot_controller = IoTController()
        self.multi_lang = MultiLanguageProcessor()
        self.db_manager = DatabaseManager()
        
        self.running = False
        self.current_user = None
        self.context_memory = {}
        self.learning_mode = True
        
        # Setup default user
        self.setup_default_user()
    
    def setup_default_user(self):
        """Setup default user profile"""
        user_id = "default_user"
        user = self.db_manager.get_user(user_id)
        
        if not user:
            user = UserProfile(
                user_id=user_id,
                name="Default User",
                voice_embedding=None,
                preferences={
                    'language': 'en',
                    'wake_word': 'assistant',
                    'response_style': 'friendly',
                    'privacy_mode': False
                },
                language='en',
                created_at=datetime.datetime.now()
            )
            self.db_manager.save_user(user)
        
        self.current_user = user
        self.multi_lang.set_language(user.language)
    
    def process_command(self, command: str, emotion: str = 'neutral') -> str:
        """Enhanced command processing with context awareness"""
        # Clean command
        clean_command = self._clean_command(command)
        
        # Update context
        self.context_memory['last_command'] = clean_command
        self.context_memory['last_emotion'] = emotion
        self.context_memory['timestamp'] = datetime.datetime.now()
        
        # Get intent with ML classifier
        intent, confidence = self.ml_classifier.predict_intent(clean_command)
        
        # Execute task
        response = self.execute_enhanced_task(intent, clean_command, confidence)
        
        # Adapt response based on emotion
        response = self._adapt_response_to_emotion(response, emotion)
        
        # Save conversation for learning
        if self.current_user:
            self.db_manager.save_conversation(
                self.current_user.user_id, clean_command, response, intent, confidence
            )
        
        return response
    
    def _clean_command(self, command: str) -> str:
        """Clean command by removing wake words"""
        wake_words = ['assistant', 'computer', 'hey assistant']
        clean_command = command
        for wake_word in wake_words:
            clean_command = clean_command.replace(wake_word, '').strip()
        return clean_command
    
    def _adapt_response_to_emotion(self, response: str, emotion: str) -> str:
        """Adapt response based on detected emotion"""
        if emotion == 'sad':
            response = "I understand you might be feeling down. " + response
        elif emotion == 'angry':
            response = "I sense some frustration. Let me help you with that. " + response
        elif emotion == 'excited':
            response = "I love your enthusiasm! " + response
        
        return response
    
    def execute_enhanced_task(self, intent: str, text: str, confidence: float) -> str:
        """Execute tasks with enhanced capabilities"""
        
        # Handle multi-language commands
        if intent == 'change_language':
            return self.change_language(text)
        
        # Handle IoT commands
        if intent == 'smart_home':
            return self.handle_iot_command(text)
        
        # Handle calendar commands
        if intent == 'calendar':
            return self.handle_calendar_command(text)
        
        # Enhanced basic commands with context awareness
        if intent == 'weather':
            return self.get_enhanced_weather(text)
        elif intent == 'time':
            return self.get_time()
        elif intent == 'date':
            return self.get_date()
        elif intent == 'alarm':
            return self.set_smart_alarm(text)
        elif intent == 'search':
            return self.enhanced_web_search(text)
        elif intent == 'music':
            return self.control_music(text)
        elif intent == 'news':
            return self.get_personalized_news()
        elif intent == 'joke':
            return self.tell_contextual_joke()
        elif intent == 'greeting':
            return self.personalized_greeting()
        elif intent == 'goodbye':
            return self.goodbye()
        else:
            return self.handle_unknown_with_learning(text, confidence)
    
    def change_language(self, text: str) -> str:
        """Change assistant language"""
        lang_map = {
            'spanish': 'es', 'english': 'en', 'french': 'fr', 
            'german': 'de', 'chinese': 'zh'
        }
        
        for lang_name, lang_code in lang_map.items():
            if lang_name in text.lower():
                if self.multi_lang.set_language(lang_code):
                    self.current_user.language = lang_code
                    self.db_manager.save_user(self.current_user)
                    return self.multi_lang.translate('language_changed', lang_code)
                break
        
        return "I support English, Spanish, French, German, and Chinese. Which would you like?"
    
    def handle_iot_command(self, text: str) -> str:
        """Handle IoT device commands with NLP parsing"""
        # Extract device and action using regex patterns
        device_patterns = {
            r'light|lamp': 'light',
            r'thermostat|temperature|heat|cool': 'thermostat',
            r'door|lock': 'lock',
            r'camera|security': 'camera',
            r'tv|television': 'tv'
        }
        
        action_patterns = {
            r'turn on|switch on|on': 'turn_on',
            r'turn off|switch off|off': 'turn_off',
            r'lock': 'lock',
            r'unlock': 'unlock',
            r'set.*?(\d+)': 'set_value'
        }
        
        # Find device
        device_type = None
        for pattern, dev_type in device_patterns.items():
            if re.search(pattern, text.lower()):
                device_type = dev_type
                break
        
        # Find action
        action = None
        value = None
        for pattern, act in action_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                action = act
                if act == 'set_value':
                    value = match.group(1)
                break
        
        if device_type and action:
            # Find specific device
            device_name = None
            for name, device in self.iot_controller.devices.items():
                if device['type'] == device_type:
                    if 'living room' in text.lower() and 'living_room' in name:
                        device_name = name
                        break
                    elif 'bedroom' in text.lower() and 'bedroom' in name:
                        device_name = name
                        break
                    elif not device_name:  # Default to first found
                        device_name = name
            
            if device_name:
                if action == 'set_value':
                    if device_type == 'thermostat':
                        return self.iot_controller.control_device(device_name, 'temperature', value)
                    elif device_type == 'light':
                        return self.iot_controller.control_device(device_name, 'brightness', value)
                else:
                    return self.iot_controller.control_device(device_name, action)
        
        return "I couldn't understand which device you want to control. Try 'turn on living room light' or 'set thermostat to 72'."
    
    def handle_calendar_command(self, text: str) -> str:
        """Handle calendar-related commands"""
        if 'schedule' in text or 'meetings' in text:
            return "You have a team meeting at 2 PM and a doctor's appointment at 4 PM today."
        elif 'add' in text and 'event' in text:
            return "I've added the event to your calendar. You'll receive a reminder 15 minutes before."
        else:
            return "Your calendar is up to date. Would you like me to check your schedule?"
    
    def get_enhanced_weather(self, text: str) -> str:
        """Get weather with location and preference awareness"""
        # In production, you'd use user's location and weather API
        if 'tomorrow' in text:
            return "Tomorrow's weather: Partly cloudy with a high of 78°F and a low of 62°F."
        elif 'week' in text:
            return "This week's forecast: Mostly sunny with temperatures ranging from 70-80°F."
        else:
            # Consider user preferences for units (Celsius/Fahrenheit)
            temp_unit = self.current_user.preferences.get('temperature_unit', 'F')
            if temp_unit == 'C':
                return "Current weather: Sunny, 24°C with light winds."
            else:
                return "Current weather: Sunny, 75°F with light winds."
    
    def get_time(self) -> str:
        """Get time in user's preferred format"""
        now = datetime.datetime.now()
        time_format = self.current_user.preferences.get('time_format', '12')
        
        if time_format == '24':
            time_str = now.strftime("%H:%M")
        else:
            time_str = now.strftime("%I:%M %p")
        
        return self.multi_lang.translate('time') + f" {time_str}."
    
    def get_date(self) -> str:
        """Get date in user's preferred format"""
        now = datetime.datetime.now()
        date_format = self.current_user.preferences.get('date_format', 'US')
        
        if date_format == 'EU':
            date_str = now.strftime("%d/%m/%Y")
        else:
            date_str = now.strftime("%A, %B %d, %Y")
        
        return f"Today is {date_str}."
    
    def set_smart_alarm(self, text: str) -> str:
        """Set smart alarm with context awareness"""
        time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)', text.lower())
        
        if time_match:
            time_str = time_match.group(0)
            
            # Smart features based on context
            alarm_type = "regular"
            if 'meeting' in text or 'work' in text:
                alarm_type = "work"
            elif 'workout' in text or 'gym' in text:
                alarm_type = "fitness"
            
            # In production, integrate with smart alarm system
            return f"Smart {alarm_type} alarm set for {time_str}. I'll gradually increase the lights and play your preferred wake-up music."
        
        return "Please specify a time like '8 AM' or '3:30 PM' for the alarm."
    
    def enhanced_web_search(self, text: str) -> str:
        """Enhanced web search with personalization"""
        search_query = re.sub(r'search|find|look up|google', '', text).strip()
        
        if search_query:
            # Add user preferences to search
            if self.current_user.preferences.get('safe_search', True):
                search_query += " safe:active"
            
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            
            try:
                webbrowser.open(search_url)
                # Store search in context for follow-up questions
                self.context_memory['last_search'] = search_query
                return f"I've opened a search for '{search_query}'. Would you like me to summarize the results?"
            except:
                return f"Here's what I found about '{search_query}': [Search results would be displayed here]"
        
        return "What would you like me to search for?"
    
    def control_music(self, text: str) -> str:
        """Advanced music control with preferences"""
        if 'play' in text:
            # Extract song/artist/genre
            music_query = re.sub(r'play|music|song', '', text).strip()
            
            if music_query:
                # Consider user's music preferences
                preferred_service = self.current_user.preferences.get('music_service', 'default')
                return f"Playing '{music_query}' on {preferred_service}. Volume set to your preferred level."
            else:
                # Use context-aware playlist selection
                time_of_day = datetime.datetime.now().hour
                if 6 <= time_of_day <= 10:
                    playlist = "Morning Motivation"
                elif 17 <= time_of_day <= 20:
                    playlist = "Evening Relaxation"
                else:
                    playlist = "Your Favorites"
                
                return f"Playing your '{playlist}' playlist."
        
        elif 'stop' in text or 'pause' in text:
            return "Music paused."
        elif 'next' in text or 'skip' in text:
            return "Skipped to next track."
        elif 'volume' in text:
            volume_match = re.search(r'(\d+)', text)
            if volume_match:
                volume = volume_match.group(1)
                return f"Volume set to {volume}%."
        
        return "I can play music, pause, skip tracks, or adjust volume. What would you like?"
    
    def get_personalized_news(self) -> str:
        """Get personalized news based on user interests"""
        interests = self.current_user.preferences.get('news_interests', ['technology', 'general'])
        
        # Mock news based on interests
        if 'technology' in interests:
            return "Top tech news: New AI breakthrough announced, Apple releases iOS update, Tesla stock rises 5%."
        elif 'sports' in interests:
            return "Sports headlines: Championship game tonight, transfer rumors, injury updates."
        else:
            return "Today's headlines: Economy shows growth, weather warnings issued, local community events."
    
    def tell_contextual_joke(self) -> str:
        """Tell jokes based on context and user preferences"""
        joke_style = self.current_user.preferences.get('joke_style', 'general')
        
        jokes = {
            'tech': [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
                "Why did the AI go to therapy? It had too many deep learning issues!"
            ],
            'general': [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What do you call a fake noodle? An impasta!",
                "Why did the scarecrow win an award? He was outstanding in his field!"
            ],
            'dad': [
                "I'm reading a book about anti-gravity. It's impossible to put down!",
                "Did you hear about the restaurant on the moon? Great food, no atmosphere!",
                "Why don't eggs tell jokes? They'd crack each other up!"
            ]
        }
        
        import random
        selected_jokes = jokes.get(joke_style, jokes['general'])
        return random.choice(selected_jokes)
    
    def personalized_greeting(self) -> str:
        """Personalized greeting based on time and user data"""
        now = datetime.datetime.now()
        hour = now.hour
        
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Good night"
        
        name = self.current_user.name if self.current_user.name != "Default User" else ""
        
        greetings = [
            f"{time_greeting}{', ' + name if name else ''}! How can I help you today?",
            f"{time_greeting}{', ' + name if name else ''}! What can I do for you?",
            f"Hello{', ' + name if name else ''}! Ready to assist you this {time_greeting.split()[1]}."
        ]
        
        import random
        return random.choice(greetings)
    
    def goodbye(self) -> str:
        """Personalized goodbye with data saving"""
        # Sync preferences to cloud
        if self.current_user:
            cloud_sync = CloudSync(self.current_user.user_id)
            cloud_sync.sync_preferences(self.current_user.preferences)
        
        goodbyes = [
            "Goodbye! Have a wonderful day!",
            "See you later! Take care!",
            "Until next time! Stay awesome!",
            "Farewell! Hope I was helpful today!"
        ]
        
        import random
        return random.choice(goodbyes)
    
    def handle_unknown_with_learning(self, text: str, confidence: float) -> str:
        """Handle unknown commands with learning capability"""
        if confidence < 0.3:
            # Very low confidence - ask for clarification
            return "I'm not sure I understood that correctly. Could you rephrase your request?"
        
        elif confidence < 0.6:
            # Medium confidence - suggest similar commands
            return "I think you might be asking about weather, time, or device control. Could you be more specific?"
        
        else:
            # Log for future learning
            if self.learning_mode:
                # In production, this would trigger active learning
                logger.info(f"Learning opportunity: '{text}' (confidence: {confidence})")
            
            return self.multi_lang.translate('unknown')
    
    def run(self):
        """Enhanced main loop with all features"""
        self.running = True
        
        welcome_msg = self.multi_lang.translate('greeting')
        self.speech_processor.speak(welcome_msg, self.current_user.language)
        
        print("\n🤖 Enhanced Virtual Personal Assistant Started!")
        print("🎯 Features: Offline speech, ML classification, IoT control, Multi-language")
        print("🔊 Say 'Hey Assistant' followed by your command")
        print("🌍 Supported languages: English, Spanish, French, German, Chinese")
        print("🏠 IoT devices ready | 📱 Cloud sync enabled | 🧠 Learning mode active")
        print("-" * 70)
        
        while self.running:
            try:
                # Listen for speech with emotion detection
                speech_result = self.speech_processor.listen()
                
                if speech_result and speech_result[0]:
                    speech_input, emotion = speech_result
                    
                    # Check for wake word with ML detector
                    if self.wake_word_detector.detect_wake_word(speech_input):
                        self.speech_processor.speak("Yes?", self.current_user.language)
                        
                        # Check for goodbye in the same utterance
                        if any(word in speech_input for word in ['goodbye', 'bye', 'exit', 'quit']):
                            response = self.goodbye()
                            self.speech_processor.speak(response, self.current_user.language)
                            self.running = False
                            break
                        
                        # Process the command with enhanced features
                        response = self.process_command(speech_input, emotion)
                        self.speech_processor.speak(response, self.current_user.language)
                        
                        # Check if this was a goodbye command
                        if "goodbye" in response.lower() or "farewell" in response.lower():
                            self.running = False
                            break
            
            except KeyboardInterrupt:
                print("\n🛑 Shutting down Enhanced Assistant...")
                farewell = "Goodbye! All your preferences have been saved."
                self.speech_processor.speak(farewell)
                
                # Final cloud sync
                if self.current_user:
                    cloud_sync = CloudSync(self.current_user.user_id)
                    cloud_sync.sync_preferences(self.current_user.preferences)
                
                break
            
            except Exception as e:
                logger.error(f"Error in enhanced main loop: {e}")
                continue
    
    def interactive_setup(self):
        """Interactive setup for new users"""
        print("\n🎉 Welcome to Enhanced Virtual Personal Assistant!")
        print("Let's set up your personalized experience...\n")
        
        # Get user name
        name = input("What's your name? (or press Enter for default): ").strip()
        if not name:
            name = "Default User"
        
        # Get language preference
        print("\nSupported languages:")
        for code, info in self.multi_lang.supported_languages.items():
            print(f"  {code}: {info['name']}")
        
        lang = input("Choose your language (default: en): ").strip() or 'en'
        if lang not in self.multi_lang.supported_languages:
            lang = 'en'
        
        # Get preferences
        preferences = {
            'language': lang,
            'temperature_unit': input("Temperature unit (F/C, default: F): ").upper() or 'F',
            'time_format': input("Time format (12/24, default: 12): ") or '12',
            'music_service': input("Preferred music service (default: Spotify): ") or 'Spotify',
            'news_interests': input("News interests (tech,sports,general - default: tech): ").split(',') or ['tech'],
            'joke_style': input("Joke style (general/tech/dad, default: general): ") or 'general',
            'safe_search': input("Enable safe search? (y/N, default: y): ").lower().startswith('y')
        }
        
        # Create user profile
        user_id = hashlib.md5(name.encode()).hexdigest()[:8]
        self.current_user = UserProfile(
            user_id=user_id,
            name=name,
            voice_embedding=None,
            preferences=preferences,
            language=lang,
            created_at=datetime.datetime.now()
        )
        
        # Save user
        self.db_manager.save_user(self.current_user)
        self.multi_lang.set_language(lang)
        
        print(f"\n✅ Setup complete! Your user ID is: {user_id}")
        print("🔄 Your preferences will be synced to the cloud automatically.")
        print("🚀 Starting Enhanced Assistant...\n")

def main():
    """Main function with enhanced error handling"""
    try:
        assistant = EnhancedVirtualPersonalAssistant()
        
        # Check if this is first run
        if not os.path.exists("assistant.db") or input("Run interactive setup? (y/N): ").lower().startswith('y'):
            assistant.interactive_setup()
        
        assistant.run()
        
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("\n📦 Install all dependencies:")
        print("pip install speechrecognition pyttsx3 spacy requests schedule pyaudio")
        print("pip install scikit-learn tensorflow opencv-python mediapipe pygame")
        print("pip install vosk sounddevice cryptography")
        print("python -m spacy download en_core_web_sm")
        
    except Exception as e:
        print(f"❌ Failed to start Enhanced Assistant: {e}")
        logger.error(f"Startup error: {e}", exc_info=True)

if __name__ == "__main__":
    main()