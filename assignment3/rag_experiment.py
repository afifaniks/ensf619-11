import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import chromadb
import pandas as pd
import pdfplumber
import torch
from chromadb.utils import embedding_functions
from codecarbon import EmissionsTracker
from transformers import AutoModelForCausalLM, AutoTokenizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""

    name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    generator_model: str = "Qwen/Qwen3-4B"


@dataclass
class QueryResult:
    """Results for a single query"""

    question: str
    answer: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    input_tokens: int
    output_tokens: int
    retrieved_chunks: List[str]
    co2_emissions: float = 0.0
    accuracy: str = "PENDING"  # TODO: evaluate manually


class DocumentProcessor:
    """Handles document loading and chunking"""

    def __init__(self, chunk_size: int, chunk_overlap: int, tokenizer):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def load_document(self, filepath: str) -> str:
        """Load document from file (supports .txt and .pdf)"""
        if filepath.endswith(".pdf"):
            return self._load_pdf(filepath)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

    def _load_pdf(self, filepath: str) -> str:
        """Load text from PDF file using pdfplumber"""
        print(f"Loading PDF: {filepath}")
        text = ""
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                if page_num % 10 == 0:
                    print(f"  Processed {page_num} pages...")
            print(f"  Total pages: {total_pages}")
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks based on token count"""
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            # Get chunk of tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))


class VectorStore:
    """ChromaDB-based vector store for embedding-based retrieval"""

    def __init__(self, embedding_model: str, collection_name: str = "documents"):
        print(f"Initializing ChromaDB with embedding model: {embedding_model}")

        self.client = chromadb.Client()

        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model, trust_remote_code=True
            )
        )
        try:
            self.client.delete_collection(name=collection_name)
        except:
            pass

        self.collection = self.client.create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

        self.chunks = []

    def add_documents(self, chunks: List[str]):
        """Add document chunks and create embeddings"""
        self.chunks = chunks
        print(f"Adding {len(chunks)} chunks to ChromaDB...")

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            self.collection.add(documents=chunks[i:batch_end], ids=ids[i:batch_end])
            if i % 500 == 0 and i > 0:
                print(f"  Processed {i}/{len(chunks)} chunks...")

        print(f"Successfully added {len(chunks)} chunks to ChromaDB")

    def retrieve(self, query: str, top_k: int) -> Tuple[List[str], float]:
        """Retrieve top-k most similar chunks using ChromaDB"""
        start_time = time.time()

        results = self.collection.query(query_texts=[query], n_results=top_k)

        retrieved_chunks = results["documents"][0] if results["documents"] else []

        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

        return retrieved_chunks, retrieval_time


class HuggingFaceLLMGenerator:
    """
    Generator using HuggingFace model with manual generation
    """

    def __init__(self, model_name: str, max_new_tokens: int = 256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading generator model: {model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully on {self.device}")

    def _build_chat_prompt(self, query: str, context: str) -> str:
        """
        Builds a proper chat-style prompt using the tokenizer's chat template.
        """
        # Format the content with context and question
        content = f"""Answer the following question based on the provided context. 
You should simply answer the question using the context and avoid adding any information not present in the context.
Keep the answer short and concise.

Context:
{context}

Question: {query}

Answer:"""

        # Use the correct chat message structure per model type
        if "gemma" in self.model_name.lower():
            messages = [{"role": "user", "content": content}]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context.",
                },
                {"role": "user", "content": content},
            ]

        # Apply chat template for correct tokenization
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt

    def generate(
        self, query: str, context_chunks: List[str]
    ) -> Tuple[str, float, int, int]:
        """
        Generate answer using HuggingFace LLM pipeline
        Returns: (answer, generation_time_ms, input_tokens, output_tokens)
        """
        start_time = time.time()

        # Build context from chunks
        context = "\n\n".join(
            [f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
        )

        # Use chat template for proper formatting
        prompt = self._build_chat_prompt(query, context)
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = 0

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with manual model.generate()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.5,
                )

            # Decode only the generated tokens (skip input)
            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            answer = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            # Count output tokens
            if answer:
                output_tokens = len(generated_tokens)
            else:
                answer = "The context does not contain sufficient information to answer this question."
                output_tokens = len(self.tokenizer.encode(answer))

        except Exception as e:
            print(f"  Warning: Generation failed: {e}")
            import traceback

            traceback.print_exc()
            answer = f"Generation error: {str(e)[:100]}"
            output_tokens = len(self.tokenizer.encode(answer))

        generation_time = (time.time() - start_time) * 1000

        return answer, generation_time, input_tokens, output_tokens


class RAGPipeline:
    """Complete RAG pipeline"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.generator = HuggingFaceLLMGenerator(config.generator_model)
        self.processor = DocumentProcessor(
            config.chunk_size, config.chunk_overlap, self.generator.tokenizer
        )

        collection_name = f"docs_{config.name.lower().replace(' ', '_')}"
        self.vector_store = VectorStore(config.embedding_model, collection_name)

    def build_index(self, document_path: str):
        """Build vector store from document"""
        print(f"\nBuilding index for: {document_path}")
        print(f"Configuration: {self.config.name}")

        text = self.processor.load_document(document_path)
        chunks = self.processor.chunk_text(text)
        print(f"Created {len(chunks)} chunks")

        self.vector_store.add_documents(chunks)

    def query(self, question: str) -> QueryResult:
        """Execute RAG pipeline for a single query"""
        tracker = EmissionsTracker()
        tracker.start()

        retrieved_chunks, retrieval_time = self.vector_store.retrieve(
            question, self.config.top_k
        )

        answer, generation_time, input_tokens, output_tokens = self.generator.generate(
            question, retrieved_chunks
        )

        co2_emissions = tracker.stop()

        total_time = retrieval_time + generation_time

        return QueryResult(
            question=question,
            answer=answer,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retrieved_chunks=retrieved_chunks,
            co2_emissions=co2_emissions,
        )


class ExperimentRunner:
    """Runs comparative experiments between configurations"""

    def __init__(self, document_path: str, questions: List[str]):
        self.document_path = document_path
        self.questions = questions
        self.results = []

    def run_experiment(self, config: RAGConfig) -> List[QueryResult]:
        """Run all queries for a given configuration"""
        print(f"\n{'='*60}")
        print(f"Running Experiment: {config.name}")
        print(f"{'='*60}")

        pipeline = RAGPipeline(config)
        pipeline.build_index(self.document_path)

        results = []
        for i, question in enumerate(self.questions, 1):
            print(f"\nQuery {i}/{len(self.questions)}: {question[:60]}...")
            result = pipeline.query(question)
            results.append(result)
            print(f"  Retrieval: {result.retrieval_time_ms:.2f}ms")
            print(f"  Generation: {result.generation_time_ms:.2f}ms")
            print(f"  Total: {result.total_time_ms:.2f}ms")
            print(f"  Input Tokens: {result.input_tokens}")
            print(f"  Output Tokens: {result.output_tokens}")
            print(f"  CO2 Emissions: {result.co2_emissions}kg")

        return results

    def save_results(self, all_results: Dict[str, List[QueryResult]], output_path: str):
        """Save results to CSV"""
        rows = []

        for config_name, results in all_results.items():
            for i, result in enumerate(results, 1):
                rows.append(
                    {
                        "Configuration": config_name,
                        "Question_ID": i,
                        "Question": result.question,
                        "Answer": result.answer,
                        "Retrieval_Time_ms": result.retrieval_time_ms,
                        "Generation_Time_ms": result.generation_time_ms,
                        "Total_Time_ms": result.total_time_ms,
                        "Input_Tokens": result.input_tokens,
                        "Output_Tokens": result.output_tokens,
                        "CO2_Emissions_kg": result.co2_emissions,
                        "Accuracy": result.accuracy,
                        "Retrieved_Chunks_Count": len(result.retrieved_chunks),
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        return df

    def generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics"""
        summary = df.groupby("Configuration").agg(
            {
                "Retrieval_Time_ms": "mean",
                "Generation_Time_ms": "mean",
                "Total_Time_ms": "mean",
                "Input_Tokens": "mean",
                "CO2_Emissions_kg": "mean",
            }
        )

        return summary


def main():
    """Main execution function"""

    baseline_config = RAGConfig(
        name="Baseline_HighContext_LowQuality",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=256,
        chunk_overlap=50,
        top_k=10,
    )

    optimized_config = RAGConfig(
        name="Optimized_LowContext_HighQuality",
        embedding_model="all-mpnet-base-v2",
        chunk_size=512,
        chunk_overlap=50,
        top_k=3,
    )

    questions = [
        "What datasets were used to evaluate Agentless?",
        "How many samples were used in greedy sampling for patch generation?",
        "How many bugs were fixed by Agentless in total on SWE-bench Lite?",
        "What are the different patch validation strategies employed by Agentless?",
        "Which baselines are used to compare performance of Agentless?",
        "What is the file level localization accuracy in Agentless?",
        "Is Agentless being adopted by any industry partner? If yes, who?",
        "Which LLM was used by Agentless for the experiments?",
        "Which embedding model Agentless use for localization?",
        "What does SBFL stand for?",
    ]

    for q in questions:
        print(q)

    document_path = "assignment3/data/agentless_paper.pdf"

    if not os.path.exists(document_path):
        print(f"ERROR: Document not found at: {document_path}")
        print("Please ensure the agentless_paper.pdf is in the data/ directory")
        return

    runner = ExperimentRunner(document_path, questions)

    all_results = {}
    all_results["Optimized"] = runner.run_experiment(optimized_config)
    all_results["Baseline"] = runner.run_experiment(baseline_config)

    output_dir = "assignment3/results"
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "rag_experiment_results.csv")
    df = runner.save_results(all_results, results_path)

    df = pd.read_csv(results_path)

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    summary = runner.generate_summary(df)
    print(summary)

    # Save summary
    summary_path = os.path.join(output_dir, "rag_experiment_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nSummary saved to: {summary_path}")

    print("Experiment Complete!")


if __name__ == "__main__":
    main()
