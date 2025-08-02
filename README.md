# Narrative Nexus

**Narrative Nexus** is an open-source Python application that creates the interactive storytelling experience of Narrative Nexus. Powered by LangChain and OpenAI, it uses Retrieval-Augmented Generation (RAG) to create immersive, dynamic adventures. With a Tkinter GUI, customizable parameters, story templates, and maturity controls, users can craft personalized narratives in genres like Fantasy, Sci-Fi, Mystery, or Horror.


## Features
- **Interactive Modes**: Use "Story," "Edit," "Continue," or "Do" to shape the narrative.
- **Story Templates**: Switch between pre-built themes for quick starts.
- **RAG Integration**: Retrieves relevant lore from a knowledge base (e.g., `fantasy_lore.txt`) for enhanced, context-aware responses.
- **Customizable Parameters**: Sliders for temperature (creativity), top_p (diversity), and retrieval k (lore chunks).
- **Maturity Controls**: Toggle "Safe" (family-friendly) or "Mature" modes.
- **UX Enhancements**: Dark/light mode, save/load games, undo actions, auto-scrolling display, and status bar.
- **Secure API Handling**: Loads OpenAI key from `.env` file.

## Tech Stack
- Python 3.x with Tkinter
- LangChain (for chains, memory, RAG)
- OpenAI API (via langchain-openai)
- FAISS for vector search
- python-dotenv for environment variables

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/narrative-nexus.git
   cd narrative-nexus
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
4. (Optional) Customize `fantasy_lore.txt` with your own lore for RAG.

## Usage
Run the application:
```
python main.py  # Assuming the script is named main.py
```
- Select a template from the "Templates" menu.
- Adjust parameters via sliders.
- Choose maturity level from the "Maturity" menu.
- Enter actions in the input field and submit to continue the story.
- Use menu options to save/load games or toggle themes.

## Configuration
- **Knowledge Base**: Edit `fantasy_lore.txt` to add custom lore. The app splits and embeds it for RAG.
- **Parameters**: Changes to sliders update the AI in real-time.
- **Extending Templates**: Modify the `story_templates` dictionary in the code to add new genres.

## Contributing
Contributions are welcome! Fork the repo, create a feature branch, and submit a pull request. For bugs or ideas, open an issue.

1. Fork the project.
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add some AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to LangChain and OpenAI for powerful tools.

For questions, contact [himankjanjire4@gmail.com]. Enjoy your adventures!
