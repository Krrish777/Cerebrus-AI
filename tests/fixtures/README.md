# Test Fixtures

This directory contains test audio files and other fixtures for testing the audio transcriber.

## harvard.wav

The Harvard sentences are a collection of phonetically-balanced sentences used in speech recognition testing. 

To use this test file, you need to place a `harvard.wav` audio file in this directory. 

You can download Harvard sentences audio from various sources:
- [Common Voice Dataset](https://commonvoice.mozilla.org/)
- [LibriSpeech](https://www.openslr.org/12)
- Record your own reading of Harvard sentences

Example Harvard sentences:
1. "The birch canoe slid on the smooth planks."
2. "Glue the sheet to the dark blue background."
3. "It's easy to tell the depth of a well."
4. "These days a chicken leg is a rare dish."
5. "Rice is often served in round bowls."

For testing purposes, any clear audio file with speech will work.

## Testing with Real Files

To run tests with actual audio files:

1. Place `harvard.wav` in this fixtures directory
2. Set your AssemblyAI API key:
   ```bash
   $env:ASSEMBLYAI_API_KEY="your_api_key_here"
   ```
3. Run the tests:
   ```bash
   pytest tests/test_audio_transcriber.py -v
   ```

## Mock Testing

The test suite includes comprehensive mocking, so it will run successfully even without real audio files or API keys.