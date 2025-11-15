# 🎙️ Agora Conversational AI Setup Guide

This guide will help you obtain the necessary credentials and configure your environment for Agora Conversational AI integration.

## 📋 Required Credentials

You need the following credentials:

1. **AGORA_APP_ID** - Your project's App ID
2. **AGORA_CUSTOMER_ID** - Customer ID for API authentication
3. **AGORA_CUSTOMER_SECRET** - Customer Secret for API authentication
4. **AGORA_RTC_TOKEN** - RTC token for channel authentication (optional but recommended)
5. **ELEVENLABS_API_KEY** - ElevenLabs API key for TTS
6. **GEMINI_API_KEY** - Google Gemini API key for LLM
7. **ARES_ASR_API_KEY** - Ares ASR API key for speech recognition (optional)
8. **ARES_ASR_REGION** - Ares ASR region (optional)

## 🔍 Step 1: Get Agora Credentials

### 1.1 Create Agora Account
1. Go to [https://console.agora.io](https://console.agora.io)
2. Sign up for a free account or log in if you already have one
3. Complete email verification

### 1.2 Create a New Project
1. In the Agora Console dashboard, click **"Create Project"**
2. Enter a project name (e.g., "Cerebrus AI Voice Assistant")
3. Select **"Use App ID + App certificate"** for authentication mode
4. Click **"Create"**
5. **Copy and save your App ID** - this is your `AGORA_APP_ID`

### 1.3 Enable Conversational AI
1. In your project dashboard, find **"Products & Features"** section
2. Look for **"Conversational AI"** and click **"Enable"**
3. Follow the setup wizard to activate the service

### 1.4 Get Customer ID and Customer Secret
1. In the left sidebar, navigate to **"Developer Toolkit"** → **"RESTful API"**
2. Click **"Add a secret"** button
3. Click **"OK"** to confirm
4. You'll see your **Customer ID** (this is your `AGORA_CUSTOMER_ID`)
5. Click **"Download"** to get the `key_and_secret.txt` file
6. Open the downloaded file to get your **Customer Secret** (this is your `AGORA_CUSTOMER_SECRET`)

⚠️ **Important**: The Customer Secret can only be downloaded once! Save it securely.

### 1.5 Generate RTC Token (Optional but Recommended)
1. Go to **"Developer Toolkit"** → **"Token Generator"**
2. Enter your channel name (e.g., "cerebrus-voice-channel")
3. Set UID to 0 (or leave empty for auto-generation)
4. Set expiration time (24 hours for testing, longer for production)
5. Click **"Generate Token"**
6. Copy the generated token - this is your `AGORA_RTC_TOKEN`

## 🎤 Step 2: Get ElevenLabs API Key

### 2.1 Create ElevenLabs Account
1. Go to [https://elevenlabs.io](https://elevenlabs.io)
2. Sign up for a free account
3. Verify your email

### 2.2 Get API Key
1. Log in to your ElevenLabs dashboard
2. Click on your profile (top right) → **"Profile + API Key"**
3. Copy your API key - this is your `ELEVENLABS_API_KEY`

### 2.3 Choose Voice (Optional)
1. Go to **"Voices"** in the dashboard
2. Browse available voices or use the default "Adam" voice
3. The voice ID is already configured in the system as `pNInz6obpgDQGcFmaJgB` (Adam)

## 🧠 Step 3: Get Google Gemini API Key

### 3.1 Create Google AI Studio Account
1. Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Accept the terms of service

### 3.2 Generate API Key
1. Click **"Create API Key"**
2. Select or create a Google Cloud project
3. Copy the generated API key - this is your `GEMINI_API_KEY`

## 🗣️ Step 4: Get Ares ASR Credentials (Optional)

Since you're using Ares as the ASR vendor, you may need to configure specific credentials if required by your Ares ASR setup. Ares ASR integration through Agora typically handles speech recognition automatically, but you can configure additional parameters if needed.

### 4.1 Configure Ares ASR (Optional)
1. If your Ares ASR setup requires an API key, add it to your environment variables
2. Set the appropriate region if required by your Ares configuration
3. The default configuration should work for most Ares ASR setups through Agora

## 🔧 Step 5: Configure Environment Variables

Create or update your `.env` file with the following:

```env
# Agora Credentials
AGORA_APP_ID=your_app_id_here
AGORA_CUSTOMER_ID=your_customer_id_here
AGORA_CUSTOMER_SECRET=your_customer_secret_here
AGORA_RTC_TOKEN=your_rtc_token_here

# ElevenLabs TTS
ELEVEN_LABS_API_KEY=sk_026cab3c1f18468eac7127990211b1ad5a04d1206d222f69
ELEVENLABS_API_KEY=sk_026cab3c1f18468eac7127990211b1ad5a04d1206d222f69

# Google Gemini LLM
GEMINI_API_KEY=AIzaSyBzZP4LGNJQdCvB3ww7Kv8cJ3RVG9crTow

# Ares ASR (Optional - for advanced configurations)
ARES_ASR_API_KEY=your_ares_api_key_if_needed
ARES_ASR_REGION=us-east-1

# Legacy environment variable names for backward compatibility
AZURE_TTS_API_KEY=sk_026cab3c1f18468eac7127990211b1ad5a04d1206d222f69
AZURE_ASR_API_KEY=your_ares_api_key_if_needed
```

## 📱 Step 6: Test Your Setup

1. Save your `.env` file
2. Restart your application
3. Run: `streamlit run main.py`
4. Navigate to the **"Voice Chat"** tab
5. Click **"🚀 Initialize Agora Conversational AI"**
6. If all credentials are correct, you should see "✅ Agora Conversational AI initialized successfully!"

## 🚨 Troubleshooting

### Common Issues:

1. **"Missing Agora credentials"**
   - Double-check your `.env` file
   - Ensure no spaces around the `=` sign
   - Restart your application after updating `.env`

2. **"Authentication failed"**
   - Verify your Customer ID and Customer Secret
   - Make sure you downloaded the secret correctly
   - Check that your project has Conversational AI enabled

3. **"ElevenLabs API error"**
   - Verify your ElevenLabs API key
   - Check your ElevenLabs account quota
   - Ensure you're not exceeding rate limits

4. **"Gemini API error"**
   - Verify your Gemini API key
   - Check that the Gemini API is enabled in Google Cloud Console
   - Ensure you have sufficient quota

5. **"Azure Speech error"**
   - Verify your Azure Speech key and region
   - Check that your Azure subscription is active
   - Ensure the Speech service is properly deployed

### Getting Help:

- Check the terminal/console for detailed error messages
- Verify all API keys are valid and not expired
- Test each service individually using their respective dashboards

## 💰 Cost Considerations

- **Agora**: Free tier includes 10,000 minutes per month
- **ElevenLabs**: Free tier includes 10,000 characters per month
- **Google Gemini**: Free tier with generous quotas
- **Azure Speech**: Free tier includes 5 audio hours per month

## 🔐 Security Best Practices

1. Never commit your `.env` file to version control
2. Use environment variables in production
3. Rotate API keys regularly
4. Monitor usage to detect unauthorized access
5. Use specific permissions for service accounts when available

---

🎉 **You're all set!** Your Cerebrus AI should now support ultra-low latency voice conversations with ElevenLabs TTS, Google Gemini LLM, and Azure Speech ASR integration.