import os
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pickle  # For saving/loading game state
from dotenv import load_dotenv  # Added for secure API key loading

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# Set up embeddings for RAG
embeddings = OpenAIEmbeddings()

# Load and prepare knowledge base for RAG (example: fantasy lore documents)
# For demonstration, assume a text file 'fantasy_lore.txt' with sample content
loader = TextLoader("fantasy_lore.txt")  # Create this file with fantasy descriptions, e.g., about dragons, castles, etc.
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Set up a prompt template with RAG context and maturity control
prompt = PromptTemplate(
    input_variables=["history", "input", "mode", "context", "template", "maturity"],
    template="""
    You are an Narrative Nexus Master creating an interactive adventure story based on the selected template: {template}.
    Keep responses engaging, descriptive, and in second person (e.g., "You see a dragon...").
    Respond to the player's actions and continue the story logically.
    Adhere to the maturity level: {maturity}.
    Use the following context from lore to enhance the story where relevant:
    
    Context: {context}
    
    Mode: {mode} (e.g., 'Story' for narration, 'Edit' for revisions, 'Continue' to advance, 'Do' for actions)
    
    Story so far: {history}
    
    Player's input: {input}
    
    Your response:
    """
)

# Set up memory to keep track of conversation history
memory = ConversationBufferMemory(input_key="input", memory_key="history")

# Define multiple story templates
story_templates = {
    "Fantasy": "A classic fantasy world with magic, dragons, and quests. You awaken in a mysterious forest. A path leads north to a castle, and south to a dark cave.",
    "Sci-Fi": "A futuristic sci-fi universe with spaceships, aliens, and technology. You wake up on a derelict spaceship drifting through space. Alarms blare as an unknown vessel approaches.",
    "Mystery": "A detective mystery in a noir city. You are a private investigator in a rainy metropolis. A shadowy figure knocks on your door with a case about a missing heirloom.",
    "Horror": "A chilling horror story with supernatural elements. You find yourself in an abandoned mansion during a storm. Strange noises echo from the attic."
}

# Define maturity levels
maturity_levels = {
    "Safe": "Keep all content family-friendly and appropriate for all ages. Avoid violence, explicit language, or mature themes.",
    "Mature": "Allow mature themes, including violence, explicit language, and adult content where appropriate to the story."
}

# GUI Setup with Enhanced UX, RAG, Story Templates, Parameter Controls, and Maturity
class AIDungeonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Narrative Nexus")
        self.root.geometry("800x600")  # Set initial window size for better UX
        
        # Theme variables (for dark/light mode)
        self.theme = "light"
        self.bg_color = "#FFFFFF"
        self.text_color = "#000000"
        self.button_bg = "#F0F0F0"
        
        # Menu bar for additional features
        self.menu = tk.Menu(root)
        root.config(menu=self.menu)
        
        # File menu: Save/Load
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Game", command=self.save_game)
        file_menu.add_command(label="Load Game", command=self.load_game)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=root.quit)
        
        # View menu: Theme toggle
        view_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Dark Mode", command=self.toggle_theme)
        
        # Template menu: Select story template
        template_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Templates", menu=template_menu)
        self.selected_template = tk.StringVar(value="Fantasy")  # Default template
        for name in story_templates.keys():
            template_menu.add_radiobutton(label=name, variable=self.selected_template, value=name, command=self.apply_template)
        
        # Maturity menu: Select maturity level
        maturity_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Maturity", menu=maturity_menu)
        self.selected_maturity = tk.StringVar(value="Safe")  # Default to Safe
        for name in maturity_levels.keys():
            maturity_menu.add_radiobutton(label=name, variable=self.selected_maturity, value=name, command=self.update_params)
        
        # Help menu
        help_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Settings frame for controlling parameters
        self.settings_frame = tk.Frame(root, bg=self.bg_color)
        self.settings_frame.pack(pady=5, fill=tk.X)
        
        # Temperature slider
        tk.Label(self.settings_frame, text="Temperature:", bg=self.bg_color, fg=self.text_color).pack(side=tk.LEFT, padx=5)
        self.temperature_var = tk.DoubleVar(value=0.7)
        temperature_slider = tk.Scale(self.settings_frame, from_=0.0, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                                      variable=self.temperature_var, command=self.update_params, bg=self.bg_color, fg=self.text_color)
        temperature_slider.pack(side=tk.LEFT, padx=5)
        
        # Top P slider
        tk.Label(self.settings_frame, text="Top P:", bg=self.bg_color, fg=self.text_color).pack(side=tk.LEFT, padx=5)
        self.top_p_var = tk.DoubleVar(value=1.0)
        top_p_slider = tk.Scale(self.settings_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                                variable=self.top_p_var, command=self.update_params, bg=self.bg_color, fg=self.text_color)
        top_p_slider.pack(side=tk.LEFT, padx=5)
        
        # Retrieval K slider
        tk.Label(self.settings_frame, text="Retrieval K:", bg=self.bg_color, fg=self.text_color).pack(side=tk.LEFT, padx=5)
        self.retrieval_k_var = tk.IntVar(value=3)
        retrieval_k_slider = tk.Scale(self.settings_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                      variable=self.retrieval_k_var, command=self.update_params, bg=self.bg_color, fg=self.text_color)
        retrieval_k_slider.pack(side=tk.LEFT, padx=5)
        
        # Story display area with auto-scroll
        self.story_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20, font=("Arial", 12),
                                                   bg=self.bg_color, fg=self.text_color)
        self.story_text.pack(pady=10, fill=tk.BOTH, expand=True)
        self.apply_template()  # Initialize with default template
        
        # Input area
        self.input_frame = tk.Frame(root, bg=self.bg_color)
        self.input_frame.pack(pady=10, fill=tk.X)
        
        self.input_entry = tk.Entry(self.input_frame, width=60, font=("Arial", 12), bg=self.button_bg, fg=self.text_color)
        self.input_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", lambda event: self.process_input())  # Enter key submits
        
        # Buttons for modes: Story, Edit, Continue, Do
        self.mode = tk.StringVar(value="Do")  # Default mode
        
        buttons = [
            ("Story", "Story"),
            ("Edit", "Edit"),
            ("Continue", "Continue"),
            ("Do", "Do")
        ]
        
        for text, mode in buttons:
            btn = tk.Button(self.input_frame, text=text, command=lambda m=mode: self.set_mode(m),
                            bg=self.button_bg, fg=self.text_color)
            btn.pack(side=tk.LEFT, padx=5)
        
        # Submit button
        submit_btn = tk.Button(self.input_frame, text="Submit", command=self.process_input,
                               bg=self.button_bg, fg=self.text_color)
        submit_btn.pack(side=tk.LEFT, padx=5)
        
        # Undo button for better UX
        undo_btn = tk.Button(self.input_frame, text="Undo", command=self.undo_last, bg=self.button_bg, fg=self.text_color)
        undo_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar for feedback
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                              bg=self.bg_color, fg=self.text_color)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize parameters and chain
        self.llm = None
        self.retriever = None
        self.rag_chain = None
        self.update_params()  # Initial setup
    
    def set_mode(self, mode):
        self.mode.set(mode)
    
    def apply_template(self):
        template_name = self.selected_template.get()
        initial_story = f"Narrative Nexus!\nTemplate: {template_name}\n{story_templates[template_name]}\nWhat do you do?\n"
        self.story_text.config(state=tk.NORMAL)
        self.story_text.delete("1.0", tk.END)
        self.story_text.insert(tk.END, initial_story)
        self.story_text.config(state=tk.DISABLED)
        memory.clear()  # Reset memory for new template
        self.status_var.set(f"Applied template: {template_name}")
    
    def update_params(self, *args):
        # Update LLM with new temperature and top_p
        self.llm = OpenAI(temperature=self.temperature_var.get(), top_p=self.top_p_var.get())
        
        # Update retriever with new k
        self.retriever = FAISS.from_documents(splits, embeddings).as_retriever(search_kwargs={"k": self.retrieval_k_var.get()})
        
        # Recreate the RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "input": RunnablePassthrough(),
                "mode": RunnablePassthrough(),
                "history": RunnablePassthrough(),
                "template": RunnablePassthrough(),
                "maturity": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        self.status_var.set("Parameters updated")
    
    def process_input(self):
        user_input = self.input_entry.get()
        if not user_input:
            return
        
        mode = self.mode.get()
        template = self.selected_template.get()
        maturity = maturity_levels[self.selected_maturity.get()]
        
        # Get current history from memory
        history = memory.load_memory_variables({})["history"]
        
        # Generate response using RAG chain
        try:
            response = self.rag_chain.invoke({"input": user_input, "mode": mode, "history": history, "template": template, "maturity": maturity})
            
            # Save to memory
            memory.save_context({"input": user_input}, {"output": response})
            
            # Update story display
            self.story_text.config(state=tk.NORMAL)
            self.story_text.insert(tk.END, f"\n> {mode}: {user_input}\n{response}\n")
            self.story_text.config(state=tk.DISABLED)
            self.story_text.see(tk.END)
            self.status_var.set("Response generated successfully with RAG")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        
        # Clear input
        self.input_entry.delete(0, tk.END)
    
    def undo_last(self):
        if memory.chat_memory.messages:
            # Remove last user input and AI response from memory
            memory.chat_memory.messages = memory.chat_memory.messages[:-2]
            # Refresh story display (truncate for simplicity)
            self.story_text.config(state=tk.NORMAL)
            self.story_text.delete("end-5l", tk.END)  # Approximate removal of last input/response
            self.story_text.config(state=tk.DISABLED)
            self.status_var.set("Undid last action")
        else:
            self.status_var.set("Nothing to undo")
    
    def save_game(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".dungeon", filetypes=[("Dungeon Save", "*.dungeon")])
        if file_path:
            save_data = {
                "memory": memory.load_memory_variables({}),
                "story": self.story_text.get("1.0", tk.END),
                "template": self.selected_template.get(),
                "temperature": self.temperature_var.get(),
                "top_p": self.top_p_var.get(),
                "retrieval_k": self.retrieval_k_var.get(),
                "maturity": self.selected_maturity.get()
            }
            with open(file_path, "wb") as f:
                pickle.dump(save_data, f)
            self.status_var.set("Game saved")
    
    def load_game(self):
        file_path = filedialog.askopenfilename(filetypes=[("Dungeon Save", "*.dungeon")])
        if file_path:
            with open(file_path, "rb") as f:
                save_data = pickle.load(f)
            memory.save_context({}, {"history": save_data["memory"]["history"]})
            self.story_text.config(state=tk.NORMAL)
            self.story_text.delete("1.0", tk.END)
            self.story_text.insert(tk.END, save_data["story"])
            self.story_text.config(state=tk.DISABLED)
            self.selected_template.set(save_data["template"])
            self.temperature_var.set(save_data.get("temperature", 0.7))
            self.top_p_var.set(save_data.get("top_p", 1.0))
            self.retrieval_k_var.set(save_data.get("retrieval_k", 3))
            self.selected_maturity.set(save_data.get("maturity", "Safe"))
            self.update_params()  # Apply loaded parameters
            self.status_var.set("Game loaded")
    
    def toggle_theme(self):
        if self.theme == "light":
            self.theme = "dark"
            self.bg_color = "#333333"
            self.text_color = "#FFFFFF"
            self.button_bg = "#555555"
        else:
            self.theme = "light"
            self.bg_color = "#FFFFFF"
            self.text_color = "#000000"
            self.button_bg = "#F0F0F0"
        
        # Update colors
        self.story_text.config(bg=self.bg_color, fg=self.text_color)
        self.settings_frame.config(bg=self.bg_color)
        self.input_frame.config(bg=self.bg_color)
        self.input_entry.config(bg=self.button_bg, fg=self.text_color)
        for child in self.input_frame.winfo_children():
            if isinstance(child, tk.Button):
                child.config(bg=self.button_bg, fg=self.text_color)
        for child in self.settings_frame.winfo_children():
            if isinstance(child, tk.Label):
                child.config(bg=self.bg_color, fg=self.text_color)
            elif isinstance(child, tk.Scale):
                child.config(bg=self.bg_color, fg=self.text_color)
        self.root.config(bg=self.bg_color)  # Note: Doesn't affect all elements perfectly
    
    def show_about(self):
        messagebox.showinfo("About", "Narrative Nexus")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = AIDungeonGUI(root)
    root.mainloop()
