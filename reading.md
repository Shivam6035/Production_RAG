### RAG Indexing Pipeline: Document Processing

1. Document Loading

* Purpose: Extract raw text from source files.

* Capability: Handle diverse file formats (e.g., PDFs, Markdown, TXT, HTML).

2. Text Splitting (Chunking)

* Chunk Size: Break text down into manageable pieces of 500–1000 characters.

* Overlap: Add an overlap of 100–200 characters between consecutive chunks to maintain contextual continuity.

* Strategy: Ensure splitting preserves sentence boundaries so meaning isn't lost mid-sentence.

3. Embedding Generation

* Purpose: Convert each processed text chunk into a high-dimensional vector representation.

* Providers: Leverage model APIs such as OpenAI, Cohere, or Google Gemini.

4. Vector Storage & Indexing

* Storage: Save the generated vector embeddings into a vector database (e.g., Chroma, Pinecone).

* Indexing: Index the vectors efficiently to enable ultra-fast similarity searches later.

Pipeline Complete

* Status: Ready for queries! The system can now accept user prompts, perform vector searches, and retrieve relevant context.