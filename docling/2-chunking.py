from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
from dotenv import load_dotenv
from openai import OpenAI



load_dotenv()

# Initialize OpenAI client (make sure you have OPENAI_API_KEY in your environment variables)
client = OpenAI()


tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length


# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------
converter = DocumentConverter()


sitemap_urls = get_sitemap_urls("https://www.underwoodlaw.net/")
conv_results_iter = converter.convert_all(sitemap_urls)

docs = []
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)
        markdown_output = document.export_to_markdown()
        json_output = document.export_to_dict()
        print(markdown_output)

result = converter.convert("https://arxiv.org/pdf/2408.09869")

document = result.document
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()
print(markdown_output)


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------




chunker = HybridChunker(
    tokenizer="allenai/longformer-base-4096",  # 4096 token limit
    max_tokens=4096,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

len(chunks)