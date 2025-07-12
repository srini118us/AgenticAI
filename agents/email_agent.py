# agents/email_agent.py - Email Notification Agent
"""
Email agent for sending notifications with Gmail and Outlook support.

This agent handles:
- Email content formatting with analysis results
- SMTP delivery with retry logic for both Gmail and Outlook
- Multiple recipient support
- Provider-specific authentication (App passwords for Gmail, regular for Outlook)
"""

import smtplib
import ssl
import time
from typing import Dict, Any, List
from datetime import datetime
from email.message import EmailMessage

from .base_agent import BaseAgent


class EmailAgent(BaseAgent):
    """
    Agent responsible for sending email notifications with Gmail/Outlook support.
    
    Key Features:
    - Both Gmail and Outlook SMTP support
    - Retry logic for reliable delivery
    - Professional email formatting
    - Analysis results integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize email agent with provider-specific configuration"""
        super().__init__("EmailAgent", config)
        
        # Detect email provider from configuration
        self.email_provider = config.get('email_provider', 'gmail').lower()
        
        # Provider-specific configuration
        if self.email_provider == 'gmail':
            self.smtp_server = 'smtp.gmail.com'
            self.smtp_port = 587
            self.email_address = config.get('gmail_email')
            self.email_password = config.get('gmail_app_password')
        elif self.email_provider == 'outlook':
            self.smtp_server = 'smtp-mail.outlook.com'
            self.smtp_port = 587
            self.email_address = config.get('outlook_email')
            self.email_password = config.get('outlook_password')
        else:
            # Fallback to Gmail configuration
            self.smtp_server = 'smtp.gmail.com'
            self.smtp_port = 587
            self.email_address = config.get('gmail_email')
            self.email_password = config.get('gmail_app_password')
        
        self.use_tls = True
        self.max_retries = config.get('email_retries', 3)
        self.retry_delay = config.get('email_retry_delay', 5.0)
        self.timeout = config.get('email_timeout', 30)
        
        # Validate configuration
        self._validate_email_config()
    
    def _validate_email_config(self):
        """Validate email configuration and log status"""
        if not self.email_address or not self.email_password:
            self.log_warning(f"{self.email_provider.title()} email credentials not configured")
            return False
        
        # Basic email format validation
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', self.email_address):
            self.log_error(f"Invalid email format: {self.email_address}")
            return False
        
        self.log_info(f"{self.email_provider.title()} email configuration validated for: {self.email_address}")
        return True
    
    def send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for sending emails.
        
        Args:
            email_data: Dict containing recipients, summary, query, and system_summaries
            
        Returns:
            Dict with email delivery status or error information
        """
        self.start_timer()
        
        try:
            self.log_info("Starting email send process")
            
            # Validate email data
            validation_result = self._validate_email_data(email_data)
            if not validation_result['valid']:
                return self.handle_error(ValueError(validation_result['error']), "Email Validation")
            
            recipients = email_data.get('recipients', [])
            if not recipients:
                return self.handle_error(ValueError("No recipients specified"), "Email Send")
            
            # Format email content
            email_content = self._format_email_content(email_data)
            
            # Send email with retry logic
            send_result = self._send_email_with_retry(recipients, email_content)
            
            processing_time = self.end_timer("email_send")
            
            if send_result['success']:
                self.log_info(f"Email sent successfully to {len(recipients)} recipients")
                return {
                    "success": True,
                    "recipients_count": len(recipients),
                    "message": f"Email sent successfully to {len(recipients)} recipients",
                    "processing_time": processing_time,
                    "email_details": {
                        "subject": email_content['subject'],
                        "recipients": recipients,
                        "timestamp": datetime.now().isoformat(),
                        "provider": self.email_provider
                    }
                }
            else:
                return self.handle_error(Exception(send_result['error']), "Email Send")
            
        except Exception as e:
            return self.handle_error(e, "Email Send")
    
    def _validate_email_data(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate email data structure and content"""
        try:
            # Check required fields
            if 'recipients' not in email_data:
                return {'valid': False, 'error': 'Missing required field: recipients'}
            
            # Validate recipients
            recipients = email_data['recipients']
            if not isinstance(recipients, list):
                return {'valid': False, 'error': 'Recipients must be a list'}
            
            if not recipients:
                return {'valid': False, 'error': 'Recipients list is empty'}
            
            # Validate email addresses
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            for recipient in recipients:
                email_addr = recipient if isinstance(recipient, str) else recipient.get('email', '')
                if not re.match(email_pattern, email_addr):
                    return {'valid': False, 'error': f'Invalid email address: {email_addr}'}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def _format_email_content(self, email_data: Dict[str, Any]) -> Dict[str, str]:
        """Format email content including subject and body"""
        try:
            # Extract data for formatting
            summary = email_data.get('summary', {})
            query = email_data.get('query', 'SAP Analysis')
            system_summaries = email_data.get('system_summaries', {})
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Format subject
            critical_count = len(summary.get('critical_findings', []))
            if critical_count > 0:
                urgency = "CRITICAL" if critical_count >= 3 else "ALERT"
                subject = f"[{urgency}] SAP EWA Analysis - {query}"
            else:
                subject = f"SAP EWA Analysis Results - {query}"
            
            # Format body
            body_parts = [
                "SAP Early Watch Analyzer - Analysis Results",
                "=" * 50,
                "",
                f"Query: {query}",
                f"Analysis Time: {timestamp}",
                f"Confidence Score: {summary.get('confidence_score', 0) * 100:.1f}%",
                f"Email Provider: {self.email_provider.title()}",
                "",
                "EXECUTIVE SUMMARY:",
                summary.get('summary', 'Analysis completed successfully'),
                ""
            ]
            
            # Add critical findings
            critical_findings = summary.get('critical_findings', [])
            body_parts.extend([
                f"CRITICAL FINDINGS ({len(critical_findings)}):",
                "-" * 30
            ])
            
            if critical_findings:
                for i, finding in enumerate(critical_findings, 1):
                    body_parts.append(f"{i}. {finding}")
            else:
                body_parts.append("✅ No critical issues found")
            
            body_parts.append("")
            
            # Add recommendations
            recommendations = summary.get('recommendations', [])
            body_parts.extend([
                f"RECOMMENDATIONS ({len(recommendations)}):",
                "-" * 30
            ])
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    body_parts.append(f"{i}. {rec}")
            else:
                body_parts.append("ℹ️ No specific recommendations at this time")
            
            # Add performance insights
            performance_insights = summary.get('performance_insights', [])
            if performance_insights:
                body_parts.extend([
                    "",
                    "PERFORMANCE INSIGHTS:",
                    "-" * 30
                ])
                for insight in performance_insights:
                    body_parts.append(f"• {insight}")
            
            # Add system summaries if available
            if system_summaries:
                body_parts.extend([
                    "",
                    "SYSTEM DETAILS:",
                    "-" * 30
                ])
                for sys_id, sys_data in system_summaries.items():
                    health = getattr(sys_data, 'overall_health', 'UNKNOWN')
                    if hasattr(health, 'value'):
                        health = health.value
                    alerts = len(getattr(sys_data, 'critical_alerts', []))
                    body_parts.append(f"• {sys_id}: {health} ({alerts} alerts)")
            
            body_parts.extend([
                "",
                "---",
                "Generated by SAP EWA Analyzer",
                f"Report generated at: {timestamp}",
                f"Sent via: {self.email_provider.title()}",
                "",
                "This is an automated analysis report. For questions or concerns,",
                "please contact your SAP BASIS team or system administrator."
            ])
            
            return {
                'subject': subject,
                'body': '\n'.join(body_parts)
            }
            
        except Exception as e:
            self.log_error(f"Error formatting email content: {e}")
            # Fallback simple format
            return {
                'subject': f"SAP Analysis Results - {email_data.get('query', 'Report')}",
                'body': f"SAP analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nPlease check the system for detailed results.\n\nSent via: {self.email_provider.title()}"
            }
    
    def _send_email_with_retry(self, recipients: List[str], email_content: Dict[str, str]) -> Dict[str, Any]:
        """Send email with retry logic for improved reliability"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.log_info(f"Email send attempt {attempt + 1}/{self.max_retries} via {self.email_provider}")
                
                result = self._send_email_smtp(recipients, email_content)
                
                if result:
                    return {'success': True}
                else:
                    raise Exception("SMTP send returned False")
                    
            except Exception as e:
                last_error = e
                self.log_warning(f"Email send attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    self.log_info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        return {
            'success': False,
            'error': f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}"
        }
    
    def _send_email_smtp(self, recipients: List[str], email_content: Dict[str, str]) -> bool:
        """Send email using SMTP with provider-specific settings"""
        try:
            # Create email message
            msg = EmailMessage()
            msg['From'] = self.email_address
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = email_content['subject']
            msg.set_content(email_content['body'])
            
            # Create secure SSL context
            context = ssl.create_default_context()
            
            # Connect and send based on provider
            self.log_info(f"Connecting to {self.smtp_server}:{self.smtp_port}")
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls(context=context)
                    self.log_info("TLS enabled")
                
                server.login(self.email_address, self.email_password)
                self.log_info(f"Authenticated with {self.email_provider}")
                
                server.send_message(msg)
                self.log_info(f"Message sent to {len(recipients)} recipients")
            
            return True
            
        except Exception as e:
            self.log_error(f"SMTP send error ({self.email_provider}): {str(e)}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """Test email connection without sending a message"""
        try:
            self.log_info(f"Testing {self.email_provider} connection...")
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls(context=context)
                
                server.login(self.email_address, self.email_password)
            
            self.log_info(f"✅ {self.email_provider.title()} connection test successful")
            return {
                "success": True,
                "message": f"{self.email_provider.title()} connection successful",
                "provider": self.email_provider,
                "server": self.smtp_server,
                "port": self.smtp_port
            }
            
        except Exception as e:
            self.log_error(f"❌ {self.email_provider.title()} connection test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": self.email_provider
            }