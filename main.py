# main.py - Simple SAP EWA Analyzer Entry Point
"""
SAP Early Watch Analyzer - Simplified Main Entry Point
"""

import streamlit as st
import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application function."""
    try:
        logger.info("🚀 Starting SAP EWA Analyzer")
        
        # Import UI components
        try:
            from ui import (
                configure_page, create_main_header, create_footer,
                create_file_upload_section, create_processing_section,
                create_system_selection_section, create_search_section,
                create_results_section, create_sidebar,
                SessionStateManager, handle_global_error,
                add_debug_section_to_app
            )
            logger.info("✅ UI components imported successfully")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import UI components: {e}")
            st.error("🚨 UI Components Missing")
            st.error(f"Import error: {e}")
            st.info("Please ensure ui/components.py exists and is properly formatted")
            return
        
        # Configure page
        configure_page()
        
        # Initialize session state
        SessionStateManager.initialize()
        
        # Create main application layout
        create_sidebar()
        create_main_header()
        
        # File upload section
        uploaded_files = create_file_upload_section()
        
        # Document processing section
        create_processing_section(uploaded_files)
        
        # System selection and search (conditional)
        if st.session_state.get('vector_store_ready', False):
            create_system_selection_section()
            
            if st.session_state.get('selected_systems', []):
                create_search_section()
        
        # Results section
        create_results_section()
        
        # Footer
        create_footer()
        
        # Debug section if enabled
        if st.session_state.get('debug_mode'):
            add_debug_section_to_app()
        
        # Advanced options if enabled
        if st.session_state.get('show_advanced_options'):
            st.markdown("---")
            st.subheader("⚙️ Advanced Configuration")
            
            with st.expander("🔧 Processing Settings", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.session_state.chunk_size = st.slider(
                        "Chunk Size", 100, 4000, st.session_state.get('chunk_size', 1000), 100
                    )
                    st.session_state.top_k = st.slider(
                        "Search Results", 1, 50, st.session_state.get('top_k', 10)
                    )
                
                with col2:
                    st.session_state.chunk_overlap = st.slider(
                        "Chunk Overlap", 0, 1000, st.session_state.get('chunk_overlap', 200), 50
                    )
                    st.session_state.temperature = st.slider(
                        "AI Temperature", 0.0, 2.0, st.session_state.get('temperature', 0.1), 0.1
                    )
        
        logger.info("✅ Application loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        
        st.error("🚨 Critical Application Error")
        st.error(f"The application failed: {str(e)}")
        
        if st.button("🔄 Reload Application"):
            st.rerun()

if __name__ == "__main__":
    main()