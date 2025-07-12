#!/usr/bin/env python3
"""
Test Enhanced Pattern Matching for SAP EWA Reports
This script demonstrates the enhanced pattern matching that should capture
hardware-related critical issues from VMW BusinessObjects report.
"""

import re
from typing import List, Tuple

def test_enhanced_pattern_matching():
    """Test the enhanced pattern matching with sample VMW BusinessObjects content"""
    
    # Sample content that might be found in VMW BusinessObjects EWA report
    sample_vmw_content = """
    SAP System ID: VMW
    SAP Product: SAP BusinessObjects BI Platform 4.0
    Status: Productive
    
    CRITICAL ISSUES:
    - Hardware resources may have been exhausted, risking performance degradation
    - Memory utilization is at 95% which may cause system instability
    - Disk space is running low on the application server
    - CPU utilization is consistently high during peak hours
    
    WARNING ISSUES:
    - Query response time is high for complex reports
    - JVM heap size configuration needs optimization
    - Adaptive Processing Server has performance issues
    - Excel Add-In parameters deviate from recommendations
    
    RECOMMENDATIONS:
    - Assess hardware resources and consider upgrades to prevent performance issues
    - Optimize JVM settings for better memory management
    - Review and adjust Excel Add-In configuration
    - Implement monitoring for hardware resource usage
    """
    
    # Enhanced pattern matching (same as in app_enhanced.py)
    def extract_enhanced_findings(content: str, system_id: str) -> Tuple[List[str], List[str]]:
        critical_findings = []
        recommendations = []
        content_lower = content.lower()
        
        # CRITICAL ISSUES - Enhanced patterns
        critical_patterns = [
            # Hardware and performance issues
            (r'hardware.*resources.*exhausted', f"🔴 [{system_id}] Hardware resources exhausted"),
            (r'hardware.*may.*exhausted', f"🔴 [{system_id}] Hardware resources may be exhausted"),
            (r'performance.*degradation', f"🔴 [{system_id}] Performance degradation detected"),
            (r'cpu.*utilization.*high', f"🔴 [{system_id}] High CPU utilization detected"),
            (r'memory.*exhausted', f"🔴 [{system_id}] Memory resources exhausted"),
            (r'disk.*space.*exhausted', f"🔴 [{system_id}] Disk space exhausted"),
            (r'memory.*utilization.*95%', f"🔴 [{system_id}] Memory utilization at 95%"),
            (r'disk.*space.*running.*low', f"🔴 [{system_id}] Disk space running low"),
            
            # Security issues
            (r'security.*vulnerability', f"🔴 [{system_id}] Security vulnerability detected"),
            (r'critical.*security.*issue', f"🔴 [{system_id}] Critical security issue detected"),
            (r'user.*critical.*authorization', f"🔴 [{system_id}] Users with critical authorizations detected"),
            (r'client.*000.*authorization', f"🔴 [{system_id}] Critical authorizations in client 000 detected"),
            
            # Database issues
            (r'database.*parameter.*not.*set', f"🔴 [{system_id}] Database parameters not set correctly"),
            (r'hana.*network.*insecure', f"🔴 [{system_id}] SAP HANA network settings insecure"),
            (r'consistency.*check.*not.*scheduled', f"🔴 [{system_id}] Consistency checks not properly scheduled"),
            
            # Software issues
            (r'software.*outdated', f"🔴 [{system_id}] SAP Software is outdated"),
            (r'support.*no.*longer.*ensured', f"🔴 [{system_id}] SAP support no longer ensured"),
            (r'version.*outdated', f"🔴 [{system_id}] Outdated software version detected"),
            
            # General critical patterns
            (r'severe.*problem', f"🔴 [{system_id}] Severe problems detected"),
            (r'critical.*issue', f"🔴 [{system_id}] Critical issues detected"),
            (r'severe.*error', f"🔴 [{system_id}] Severe errors detected"),
            (r'fatal.*error', f"🔴 [{system_id}] Fatal errors detected"),
        ]
        
        # WARNING ISSUES - Enhanced patterns
        warning_patterns = [
            # Performance warnings
            (r'performance.*issue', f"🟡 [{system_id}] Performance issues detected"),
            (r'response.*time.*high', f"🟡 [{system_id}] High response times detected"),
            (r'query.*response.*time.*high', f"🟡 [{system_id}] Query response time is high"),
            (r'commit.*response.*time.*high', f"🟡 [{system_id}] Commit response time is high"),
            
            # Configuration warnings
            (r'parameter.*deviate', f"🟡 [{system_id}] Parameters deviate from recommendations"),
            (r'configuration.*issue', f"🟡 [{system_id}] Configuration issues detected"),
            (r'excel.*add.*in.*deviate', f"🟡 [{system_id}] Excel Add-In configuration issues"),
            (r'purge.*scheduling.*not.*conform', f"🟡 [{system_id}] Purge job scheduling issues"),
            
            # JVM and application warnings
            (r'jvm.*issue', f"🟡 [{system_id}] JVM issues detected"),
            (r'jvm.*heap.*size', f"🟡 [{system_id}] JVM heap size configuration issues"),
            (r'adaptive.*processing.*server.*issue', f"🟡 [{system_id}] Adaptive Processing Server issues"),
            (r'application.*server.*issue', f"🟡 [{system_id}] Application server issues detected"),
            
            # Data volume warnings
            (r'data.*volume.*reduction', f"🟡 [{system_id}] Data volume reduction potential identified"),
            (r'noticeable.*potential.*reduction', f"🟡 [{system_id}] Potential for data volume reduction"),
            
            # General warning patterns
            (r'warning.*issue', f"🟡 [{system_id}] Warning issues detected"),
            (r'potential.*problem', f"🟡 [{system_id}] Potential problems detected"),
        ]
        
        # Extract critical findings
        for pattern, message in critical_patterns:
            if re.search(pattern, content_lower):
                if message not in critical_findings:
                    critical_findings.append(message)
        
        # Extract warning findings
        for pattern, message in warning_patterns:
            if re.search(pattern, content_lower):
                if message not in critical_findings:  # Add to critical_findings list (contains both)
                    critical_findings.append(message)
        
        # Extract recommendations
        # Look for explicit recommendations
        if 'recommendation' in content_lower:
            # Extract recommendation sentences
            sentences = re.split(r'[.!?]', content)
            for sentence in sentences:
                if 'recommendation' in sentence.lower() and len(sentence.strip()) > 20:
                    rec = f"📋 [{system_id}] {sentence.strip()}"
                    if rec not in recommendations:
                        recommendations.append(rec)
        
        # Look for SAP Notes
        sap_note_matches = re.finditer(r'sap note (\d+)', content_lower)
        for match in sap_note_matches:
            note_num = match.group(1)
            rec = f"📋 [{system_id}] Refer to SAP Note {note_num}"
            if rec not in recommendations:
                recommendations.append(rec)
        
        # Look for specific action items
        action_patterns = [
            (r'evaluate.*secure.*hana.*network', f"📋 [{system_id}] Evaluate and secure SAP HANA network settings"),
            (r'update.*sap.*software', f"📋 [{system_id}] Update SAP Software to ensure continued support"),
            (r'review.*user.*authorization', f"📋 [{system_id}] Review user authorizations and limit critical access"),
            (r'assess.*hardware.*resource', f"📋 [{system_id}] Assess hardware resources and consider upgrades"),
            (r'adjust.*hana.*parameter', f"📋 [{system_id}] Adjust SAP HANA database parameters"),
            (r'implement.*consistency.*check', f"📋 [{system_id}] Implement global consistency checks"),
            (r'optimize.*jvm.*setting', f"📋 [{system_id}] Optimize JVM settings for better memory management"),
            (r'review.*excel.*add.*in', f"📋 [{system_id}] Review and adjust Excel Add-In configuration"),
            (r'implement.*monitoring.*hardware', f"📋 [{system_id}] Implement monitoring for hardware resource usage"),
        ]
        
        for pattern, message in action_patterns:
            if re.search(pattern, content_lower):
                if message not in recommendations:
                    recommendations.append(message)
        
        # Defaults if nothing found
        if not critical_findings:
            critical_findings = [f"ℹ️ [{system_id}] No critical issues detected"]
        if not recommendations:
            recommendations = [f"📋 [{system_id}] Regular monitoring recommended"]
        
        return critical_findings, recommendations
    
    # Test with VMW system
    print("=== ENHANCED PATTERN MATCHING TEST ===")
    print("Testing VMW BusinessObjects system...")
    print()
    
    critical_findings, recommendations = extract_enhanced_findings(sample_vmw_content, "VMW")
    
    print("🔴 CRITICAL ISSUES for VMW:")
    for finding in critical_findings:
        if "🔴" in finding:
            print(f"  {finding}")
    
    print()
    print("🟡 WARNING ISSUES for VMW:")
    for finding in critical_findings:
        if "🟡" in finding:
            print(f"  {finding}")
    
    print()
    print("📋 SAP RECOMMENDATIONS for VMW:")
    for rec in recommendations:
        print(f"  {rec}")
    
    print()
    print("=== SUMMARY ===")
    print(f"✅ Found {len([f for f in critical_findings if '🔴' in f])} critical issues")
    print(f"⚠️ Found {len([f for f in critical_findings if '🟡' in f])} warning issues")
    print(f"📋 Found {len(recommendations)} recommendations")
    
    # Test with PR0 system (IBP)
    print()
    print("=== TESTING PR0 IBP SYSTEM ===")
    
    sample_pr0_content = """
    SAP System ID: PR0
    SAP Product: SAP IBP OD 2111
    Status: Productive
    
    CRITICAL ISSUES:
    - SAP HANA network settings for System Replication are insecure
    - SAP Software on this system is outdated; support with SAP Security Notes is no longer ensured
    - Users with critical authorizations that allow actions in client 000 and other clients
    - Hardware resources may have been exhausted, risking performance degradation
    
    WARNING ISSUES:
    - Noticeable potential for reduction of data volume was identified
    - SAP HANA database parameters are not set in accordance with recommendations
    - Consistency checks for the SAP HANA database are scheduled without the global consistency check
    
    RECOMMENDATIONS:
    - Evaluate and secure SAP HANA network settings for System Replication
    - Update SAP Software to ensure continued support with SAP Security Notes
    - Review user authorizations and limit critical access where possible
    - Assess hardware resources and consider upgrades to prevent performance issues
    - Adjust SAP HANA database parameters to align with SAP recommendations
    - Implement global consistency checks for the SAP HANA database
    """
    
    critical_findings_pr0, recommendations_pr0 = extract_enhanced_findings(sample_pr0_content, "PR0")
    
    print("🔴 CRITICAL ISSUES for PR0:")
    for finding in critical_findings_pr0:
        if "🔴" in finding:
            print(f"  {finding}")
    
    print()
    print("🟡 WARNING ISSUES for PR0:")
    for finding in critical_findings_pr0:
        if "🟡" in finding:
            print(f"  {finding}")
    
    print()
    print("📋 SAP RECOMMENDATIONS for PR0:")
    for rec in recommendations_pr0:
        print(f"  {rec}")

if __name__ == "__main__":
    test_enhanced_pattern_matching() 