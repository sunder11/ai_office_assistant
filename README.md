AI OFFICE ASSITANT (CLAUDIA):

This Python Program Provides a reasonably good starting point to make an AI Office Assistant using docling to upload .docx, .pdf, .and html files to the persistent
Vectorstores(langchain)/SQLite database. It also has the functionality to enter a link to a website's sitemap.xml and process all of the pages on the website into the database.<br>

Some features of the UI that runs in your browser include a button to delete all of the data in the database, a button to delete just the chat history, 6 fields to enter the path to your files directorys (.docx-word, pdf, html, .pptx-powerpoint, .xlsx-excel, .xml-sitemap link)
with 6 separate buttons to upload each type of file to the database, and button to download the vector database and a button to download the SQLite database.<br>

By default it is setup to use llama-3.3-70b-versatile with a chunk sizes based on document type and the UI runs at http://localhost:8501. Claude wrote the code and did the fine tuning I just did what he said. 

TO USE:

Clone the Repository, Install the dependencies in Requirements.txt, Add Your GROQ_API_KEY="" to the .env file, and in the terminal run:<br>
streamlit run /home/your_user_name/your_projects_folder/ai_office_assistant/docling/ai_office_assistant.py<br><br>

Note: the files with the my.prefix are tests for each type of file format that are not persistent. The my.html.sp.py file is persistent but it only alows you to process a sitemap. They all have UI's<br>

**You can find some great batch utilities to pre-prepare your data at my other repository:**<br>
https://github.com/sunder11/Chunklings<br>
You can batch convert, .wpd => .docx, copy all your pdf files in your myfiles folder/subfolders to one folder without overwriting any files with the same name
and export your outlook.pst and outlook.ost files into individual html emails to be process by docling.<br>

Thanks to claude-opus-4-20250514 for writing most of the 2000 lines of python code for about $35.00. It would have cost a lot less but I just started learning about AI and python about 2 weeks ago.I assume this will be helpful to someone but I really do not know. it seems to suit my needs. It works locally and I have yet to learn out to deploy in production.<br>

NOTE: Processing the PDF files is very slow if you are using the CPU ( I have an AMD Ryzen 7 3700x (8 cores) and 32g ram), but it works. Some kind of NViDIA GPU is probably a good Idea (RTX 3090 24g maybe).   
