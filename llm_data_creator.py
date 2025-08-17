#!/usr/bin/env python3
"""
LLM-based Data Creation System for InferDoc

This module provides functionality to generate synthetic training data
using Large Language Models (LLM) APIs for document inference tasks.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime


class LLMDataCreator:
    """
    Creates synthetic training data using LLM APIs for document inference tasks.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize LLM Data Creator.
        
        Args:
            api_key: OpenAI API key (can be set via OPENAI_API_KEY env var)
            model: LLM model to use for data generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_document_qa_pairs(self, document_type: str, num_pairs: int = 10) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs for document inference training.
        
        Args:
            document_type: Type of document (e.g., "legal", "medical", "financial")
            num_pairs: Number of QA pairs to generate
            
        Returns:
            List of dictionaries containing question-answer pairs
        """
        prompt = f"""
Generate {num_pairs} realistic question-answer pairs for document inference training.
Document type: {document_type}

Create questions that would require understanding and inference from {document_type} documents.
Include questions about:
1. Key information extraction
2. Document classification
3. Relationship identification
4. Content summarization
5. Logical inference

Format as JSON array:
[
    {{
        "question": "What is the main purpose of this document?",
        "answer": "The document is a contract agreement...",
        "document_type": "{document_type}",
        "inference_type": "classification"
    }}
]
"""
        
        try:
            response = self._call_llm_api(prompt)
            # Parse JSON response
            qa_pairs = json.loads(response)
            
            # Add metadata
            for pair in qa_pairs:
                pair["created_at"] = datetime.now().isoformat()
                pair["model_used"] = self.model
            
            self.logger.info(f"Generated {len(qa_pairs)} QA pairs for {document_type} documents")
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"Error generating QA pairs: {e}")
            return []
    
    def generate_synthetic_documents(self, document_type: str, num_docs: int = 5) -> List[Dict[str, str]]:
        """
        Generate synthetic documents for training data.
        
        Args:
            document_type: Type of document to generate
            num_docs: Number of documents to generate
            
        Returns:
            List of synthetic documents with metadata
        """
        prompt = f"""
Generate {num_docs} realistic synthetic {document_type} documents for machine learning training.

Each document should:
1. Be realistic and contain typical information found in {document_type} documents
2. Include relevant entities, dates, numbers, and relationships
3. Have varied structure and content
4. Be suitable for document inference tasks

Format as JSON array:
[
    {{
        "content": "Full document text here...",
        "title": "Document title",
        "document_type": "{document_type}",
        "key_entities": ["entity1", "entity2"],
        "summary": "Brief summary of the document"
    }}
]
"""
        
        try:
            response = self._call_llm_api(prompt)
            documents = json.loads(response)
            
            # Add metadata
            for doc in documents:
                doc["created_at"] = datetime.now().isoformat()
                doc["model_used"] = self.model
                doc["synthetic"] = True
            
            self.logger.info(f"Generated {len(documents)} synthetic {document_type} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error generating documents: {e}")
            return []
    
    def create_training_dataset(self, document_types: List[str], output_file: str) -> Dict[str, Any]:
        """
        Create a comprehensive training dataset with documents and QA pairs.
        
        Args:
            document_types: List of document types to generate data for
            output_file: Path to save the dataset
            
        Returns:
            Dataset statistics
        """
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "model_used": self.model,
                "document_types": document_types
            },
            "documents": [],
            "qa_pairs": []
        }
        
        stats = {"total_documents": 0, "total_qa_pairs": 0, "document_types": {}}
        
        for doc_type in document_types:
            self.logger.info(f"Generating data for {doc_type} documents...")
            
            # Generate synthetic documents
            documents = self.generate_synthetic_documents(doc_type, num_docs=3)
            dataset["documents"].extend(documents)
            
            # Generate QA pairs
            qa_pairs = self.generate_document_qa_pairs(doc_type, num_pairs=8)
            dataset["qa_pairs"].extend(qa_pairs)
            
            # Update statistics
            stats["document_types"][doc_type] = {
                "documents": len(documents),
                "qa_pairs": len(qa_pairs)
            }
            stats["total_documents"] += len(documents)
            stats["total_qa_pairs"] += len(qa_pairs)
        
        # Save dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Dataset saved to {output_file}")
        self.logger.info(f"Total: {stats['total_documents']} documents, {stats['total_qa_pairs']} QA pairs")
        
        return stats
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: Prompt to send to the LLM
            
        Returns:
            LLM response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3000,
            "temperature": 0.7
        }
        
        response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


def main():
    """Example usage of LLMDataCreator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data using LLM")
    parser.add_argument("--document-types", nargs="+", default=["legal", "medical", "financial"],
                       help="Types of documents to generate")
    parser.add_argument("--output", default="synthetic_training_data.json",
                       help="Output file for the dataset")
    parser.add_argument("--model", default="gpt-3.5-turbo",
                       help="LLM model to use")
    
    args = parser.parse_args()
    
    try:
        creator = LLMDataCreator(model=args.model)
        stats = creator.create_training_dataset(args.document_types, args.output)
        
        print("\n" + "="*50)
        print("DATASET CREATION COMPLETE")
        print("="*50)
        print(f"Output file: {args.output}")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total QA pairs: {stats['total_qa_pairs']}")
        print("\nDocument type breakdown:")
        for doc_type, counts in stats['document_types'].items():
            print(f"  {doc_type}: {counts['documents']} docs, {counts['qa_pairs']} QA pairs")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
