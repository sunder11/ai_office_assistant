from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

# ----------------------------------
# covert pdf to text file.
# ----------------------------------
converter = DocumentConverter()

source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
result = converter.convert(source)

# Print results to console
print(result.document.export_to_markdown())

print(result.document.export_to_dict())  # export to JSON

# Save results to a text file
with open("doclingtext.txt", "w", encoding="utf-8") as f:
    f.write("\n\nMarkdown Conversion:\n")
    f.write(result.document.export_to_markdown())


# --------------------------------------------------------------
# Basic PDF extraction
# -------------------------------------------------------------

result = converter.convert("https://arxiv.org/pdf/2408.09869")

document = result.document
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()

print(markdown_output)

# --------------------------------------------------------------
# Basic HTML extraction
# --------------------------------------------------------------

result = converter.convert("https://www.underwoodlaw.net/")

document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)

# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------

sitemap_urls = get_sitemap_urls("https://www.underwoodlaw.net/")
conv_results_iter = converter.convert_all(sitemap_urls)

docs = []
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)
        markdown_output = document.export_to_markdown()
        print(markdown_output)
