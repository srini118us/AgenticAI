#!/usr/bin/env python3
"""
Extract sample content from EWA PDFs to understand structure
"""

import PyPDF2
import re

def extract_sample_content(file_path: str, max_chars: int = 2000):
    """Extract sample content from PDF"""
    print(f"\n📄 SAMPLE CONTENT FROM: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            content = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    content += page_text + "\n"
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return
    
    # Show first portion
    print("FIRST 2000 CHARACTERS:")
    print("-" * 40)
    print(content[:max_chars])
    print("-" * 40)
    
    # Look for specific patterns
    print("\n🔍 PATTERN ANALYSIS:")
    
    # Traffic light patterns
    traffic_patterns = [
        r'red.*rating|rating.*red',
        r'yellow.*rating|rating.*yellow', 
        r'green.*rating|rating.*green',
        r'critical.*rating',
        r'warning.*rating',
        r'healthy.*rating'
    ]
    
    for pattern in traffic_patterns:
        matches = re.findall(pattern, content.lower())
        if matches:
            print(f"🎯 {pattern}: {len(matches)} matches")
            for match in matches[:3]:  # Show first 3
                print(f"   - {match}")
    
    # System ID patterns
    system_patterns = [
        r'\bSYSTEM\s+([A-Z0-9]{2,6})\b',
        r'\bSID\s*[:=]?\s*([A-Z0-9]{2,6})\b',
        r'\[([A-Z0-9]{2,6})\]',
        r'System ID:\s*([A-Z0-9]{2,6})',
    ]
    
    print("\n🆔 SYSTEM ID PATTERNS:")
    for pattern in system_patterns:
        matches = re.findall(pattern, content.upper())
        if matches:
            print(f"   {pattern}: {matches}")
    
    # Recommendation patterns
    rec_patterns = [
        r'recommendation[s]?.*[.!]',
        r'sap.*note.*[0-9]+',
        r'action.*required',
        r'immediate.*action'
    ]
    
    print("\n📋 RECOMMENDATION PATTERNS:")
    for pattern in rec_patterns:
        matches = re.findall(pattern, content.lower())
        if matches:
            print(f"   {pattern}: {len(matches)} matches")
            for match in matches[:2]:  # Show first 2
                print(f"   - {match}")

# Analyze each PDF
files = [
    'sample-ewa-ibp.pdf',
    'ewa-busiinessobjects-sample-report.pdf',
    'earlywatch-alert-s4hana-security-chapter.pdf', 
    'ewa-ecc_production.pdf'
]

for file_path in files:
    extract_sample_content(file_path) 