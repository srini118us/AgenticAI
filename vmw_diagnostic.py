#!/usr/bin/env python3
"""
Universal SAP System Diagnostic Tool
Analyzes all PDFs for system IDs and BusinessObjects/BO/VMW mentions
"""

import re
import os
from typing import List, Dict, Any

def analyze_pdf_content(file_path: str):
    print(f"\n🔍 ANALYZING: {file_path}")
    print("=" * 60)
    try:
        import PyPDF2
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
    print(f"📊 Content length: {len(content)} characters\n")

    # Extract all system IDs
    system_patterns = [
        r'\bSYSTEM\s+([A-Z0-9]{2,6})\b',
        r'\bSID\s*[:=]?\s*([A-Z0-9]{2,6})\b',
        r'\[([A-Z0-9]{2,6})\]',
        r'System ID:\s*([A-Z0-9]{2,6})',
        r'System:\s*([A-Z0-9]{2,6})',
        r'\b([A-Z]{2,6})\s+system\b',
    ]
    found_ids = set()
    for pattern in system_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            if len(match) >= 2 and match.upper() not in ['THE', 'AND', 'FOR', 'SAP', 'EWA', 'CPU', 'RAM', 'SID', 'SYSTEM']:
                found_ids.add(match.upper())
    print(f"🆔 System IDs found: {sorted(found_ids) if found_ids else 'None'}\n")

    # Look for BusinessObjects/BO/VMW mentions
    bo_aliases = [
        (r'businessobjects', 'BusinessObjects'),
        (r'\bbo\b', 'BO'),
        (r'\bvmw\b', 'VMW'),
        (r'crystal.*reports', 'Crystal Reports'),
        (r'web.*intelligence', 'Web Intelligence'),
        (r'bi.*platform', 'BI Platform'),
    ]
    for pattern, label in bo_aliases:
        matches = re.findall(pattern, content, re.IGNORECASE)
        print(f"{label:20}: {len(matches)} matches")
    print()
    print("🔍 CONTENT SAMPLE (first 1000 chars):")
    print("-" * 50)
    print(content[:1000])
    print("-" * 50)
    print("\n✅ Analysis complete!\n")

def main():
    print("Universal SAP System Diagnostic Tool")
    print("=" * 40)
    pdfs = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if not pdfs:
        print("❌ No PDF files found in current directory")
        return
    print(f"Found {len(pdfs)} PDF(s):")
    for f in pdfs:
        print(f"  - {f}")
    for f in pdfs:
        analyze_pdf_content(f)

if __name__ == "__main__":
    main() 