#!/usr/bin/env python3
"""
Analyze EWA PDF patterns to understand common structure and improve text retrieval
"""

import PyPDF2
import re
import os
from typing import Dict, List, Set

def analyze_pdf_structure(file_path: str) -> Dict:
    """Analyze the structure and patterns in an EWA PDF"""
    print(f"\n🔍 ANALYZING: {file_path}")
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
        return {}
    
    print(f"📊 Content length: {len(content)} characters")
    print(f"📄 Pages: {len(pdf_reader.pages)}")
    
    # Analyze common EWA patterns
    analysis = {
        'file': file_path,
        'content_length': len(content),
        'pages': len(pdf_reader.pages),
        'traffic_lights': {},
        'system_ids': set(),
        'rating_patterns': [],
        'recommendation_patterns': [],
        'critical_indicators': [],
        'warning_indicators': [],
        'sap_notes': [],
        'tables_detected': False,
        'charts_detected': False
    }
    
    content_lower = content.lower()
    content_upper = content.upper()
    
    # 1. Traffic Light Analysis
    traffic_light_patterns = [
        (r'red.*rating|rating.*red|🔴|red.*circle|critical.*red', 'RED'),
        (r'yellow.*rating|rating.*yellow|🟡|yellow.*circle|warning.*yellow', 'YELLOW'),
        (r'green.*rating|rating.*green|✅|green.*circle|healthy.*green', 'GREEN'),
        (r'orange.*rating|rating.*orange|🟠|orange.*circle', 'ORANGE'),
    ]
    
    for pattern, color in traffic_light_patterns:
        matches = re.findall(pattern, content_lower)
        if matches:
            analysis['traffic_lights'][color] = len(matches)
            print(f"🎯 {color} ratings found: {len(matches)}")
    
    # 2. System ID Detection
    system_patterns = [
        r'\bSYSTEM\s+([A-Z0-9]{2,6})\b',
        r'\bSID\s*[:=]?\s*([A-Z0-9]{2,6})\b',
        r'\[([A-Z0-9]{2,6})\]',
        r'System ID:\s*([A-Z0-9]{2,6})',
        r'System:\s*([A-Z0-9]{2,6})',
        r'\b([A-Z]{2,6})\s+system\b',
        r'\b([A-Z]{1,3}[0-9]{1,2})\b',  # P01, D1, Q1, etc.
    ]
    
    for pattern in system_patterns:
        matches = re.findall(pattern, content_upper)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            if len(match) >= 2 and match not in ['THE', 'AND', 'FOR', 'SAP', 'EWA', 'CPU', 'RAM']:
                analysis['system_ids'].add(match)
    
    print(f"🆔 System IDs found: {sorted(analysis['system_ids']) if analysis['system_ids'] else 'None'}")
    
    # 3. Rating Patterns
    rating_patterns = [
        r'rating.*[0-9]+%',
        r'[0-9]+%.*rating',
        r'utilization.*[0-9]+%',
        r'[0-9]+%.*utilization',
        r'performance.*rating',
        r'rating.*performance',
    ]
    
    for pattern in rating_patterns:
        matches = re.findall(pattern, content_lower)
        if matches:
            analysis['rating_patterns'].extend(matches[:5])  # Limit to first 5
    
    # 4. Recommendation Patterns
    rec_patterns = [
        r'recommendation[s]?.*[.!]',
        r'sap.*note.*[0-9]+',
        r'action.*required',
        r'immediate.*action',
        r'consider.*upgrading',
        r'review.*configuration',
    ]
    
    for pattern in rec_patterns:
        matches = re.findall(pattern, content_lower)
        if matches:
            analysis['recommendation_patterns'].extend(matches[:5])
    
    # 5. Critical Indicators
    critical_patterns = [
        r'critical.*issue',
        r'severe.*problem',
        r'hardware.*exhausted',
        r'cpu.*9[0-9]%',
        r'memory.*9[0-9]%',
        r'disk.*full',
        r'connection.*failed',
        r'service.*failed',
        r'security.*vulnerability',
    ]
    
    for pattern in critical_patterns:
        matches = re.findall(pattern, content_lower)
        if matches:
            analysis['critical_indicators'].extend(matches[:5])
    
    # 6. Warning Indicators
    warning_patterns = [
        r'warning.*issue',
        r'attention.*required',
        r'performance.*degradation',
        r'parameters.*deviate',
        r'outdated.*version',
        r'configuration.*suboptimal',
    ]
    
    for pattern in warning_patterns:
        matches = re.findall(pattern, content_lower)
        if matches:
            analysis['warning_indicators'].extend(matches[:5])
    
    # 7. SAP Notes
    sap_note_matches = re.findall(r'sap.*note.*([0-9]+)', content_lower)
    analysis['sap_notes'] = list(set(sap_note_matches))
    
    # 8. Table/Chart Detection
    if '|' in content or '\t' in content:
        analysis['tables_detected'] = True
    if 'chart' in content_lower or 'graph' in content_lower:
        analysis['charts_detected'] = True
    
    # Print summary
    print(f"\n📋 SUMMARY for {os.path.basename(file_path)}:")
    print(f"   Traffic Lights: {analysis['traffic_lights']}")
    print(f"   SAP Notes: {analysis['sap_notes']}")
    print(f"   Tables: {'Yes' if analysis['tables_detected'] else 'No'}")
    print(f"   Charts: {'Yes' if analysis['charts_detected'] else 'No'}")
    
    return analysis

def analyze_all_ewa_files():
    """Analyze all EWA PDF files"""
    ewa_files = [
        'sample-ewa-ibp.pdf',
        'ewa-busiinessobjects-sample-report.pdf', 
        'earlywatch-alert-s4hana-security-chapter.pdf',
        'ewa-ecc_production.pdf'
    ]
    
    all_analyses = []
    common_patterns = {
        'traffic_light_colors': set(),
        'system_id_patterns': set(),
        'rating_formats': set(),
        'recommendation_keywords': set(),
        'critical_keywords': set(),
        'warning_keywords': set(),
        'sap_note_format': set()
    }
    
    for file_path in ewa_files:
        if os.path.exists(file_path):
            analysis = analyze_pdf_structure(file_path)
            if analysis:
                all_analyses.append(analysis)
                
                # Collect common patterns
                common_patterns['traffic_light_colors'].update(analysis['traffic_lights'].keys())
                common_patterns['system_id_patterns'].update(analysis['system_ids'])
                common_patterns['rating_formats'].update(analysis['rating_patterns'])
                common_patterns['recommendation_keywords'].update(analysis['recommendation_patterns'])
                common_patterns['critical_keywords'].update(analysis['critical_indicators'])
                common_patterns['warning_keywords'].update(analysis['warning_indicators'])
                common_patterns['sap_note_format'].update(analysis['sap_notes'])
    
    # Print common patterns across all files
    print("\n" + "="*80)
    print("🎯 COMMON PATTERNS ACROSS ALL EWA FILES")
    print("="*80)
    
    print(f"\n🚦 Traffic Light Colors: {sorted(common_patterns['traffic_light_colors'])}")
    print(f"🆔 System ID Patterns: {sorted(common_patterns['system_id_patterns'])}")
    print(f"📊 Rating Formats: {list(common_patterns['rating_formats'])[:10]}")  # Limit display
    print(f"📋 Recommendation Keywords: {list(common_patterns['recommendation_keywords'])[:10]}")
    print(f"🔴 Critical Keywords: {list(common_patterns['critical_keywords'])[:10]}")
    print(f"🟡 Warning Keywords: {list(common_patterns['warning_keywords'])[:10]}")
    print(f"📝 SAP Note Format: {sorted(common_patterns['sap_note_format'])}")
    
    return all_analyses, common_patterns

if __name__ == "__main__":
    analyses, patterns = analyze_all_ewa_files() 