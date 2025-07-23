from langchain_community.embeddings import OllamaEmbeddings

#pull pllama in your computer.after downloading it in your pc.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print(embeddings.embed_query("Hello world"))
