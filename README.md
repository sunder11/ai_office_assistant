AI OFFICE ASSITANT (CLAUDIA):

This Python Program Provides a reasonably good starting point to make an AI Office Assistant using docling to upload .docx, .pdf, .and html files to the persistent
Vectorstores(langchain)/SQLite database. It also has the functionality to enter a link to a website's sitemap.xml and process all of the pages on the website into the database.<br>

Some features of the UI that runs in your browser include a button to delete all of the data in the database, a button to delete just the chat history, 4 fields to enter the path to your files directorys (docx, pdf, html, sitemap link)
with 4 separate buttons to upload each type of file to the database, and button to download the vector database and a button to download the SQLite database.<br>

By default it is setup to use llama-3.3-70b-versatile with a chunk size of 1000 and overlap of 200 and the UI runs at http://localhost:8501


TO USE:

Clone the Repository, Install the dependencies in Requirements.txt, Add Your GROQ_API_KEY="" to the .env file, and in the terminal run:<br>
streamlit run /home/your_user_name/your_projects_folder/ai_office_assistant/docling/ai_office_assistant.py<br>

You can fins some great batch utilities to pre-prepare your data at my other repository:<br>
