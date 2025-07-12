# ui/__init__.py - Simple version
import logging
logger = logging.getLogger(__name__)

from .core_components import (
    configure_page,
    create_main_header,
    create_footer,
    create_file_upload_section,
    create_processing_section,
    create_system_selection_section,
    create_search_section,
    create_results_section,
    create_sidebar,
    SessionStateManager,
    handle_global_error,
    add_debug_section_to_app
)

logger.info("UI components loaded")