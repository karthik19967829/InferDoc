#!/usr/bin/env python3
"""
Example script showing how to use the LLM data creator with InferDoc
"""

from llm_data_creator import LLMDataCreator
import os

def create_inferdoc_training_data():
    """
    Create training data specifically for InferDoc tasks
    """
    # Document types relevant to InferDoc
    document_types = [
        "research_paper", 
        "technical_report", 
        "legal_document", 
        "medical_record",
        "business_contract"
    ]
    
    creator = LLMDataCreator()
    
    # Create comprehensive dataset
    stats = creator.create_training_dataset(
        document_types=document_types,
        output_file="inferdoc_training_data.json"
    )
    
    print("Created InferDoc training dataset!")
    print(f"Generated {stats['total_documents']} documents and {stats['total_qa_pairs']} QA pairs")

if __name__ == "__main__":
    create_inferdoc_training_data()
