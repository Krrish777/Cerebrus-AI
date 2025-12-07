"""
Prompt template utilities.
Handles loading and rendering prompt templates.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from jinja2 import Template, Environment, FileSystemLoader, TemplateNotFound

from src.core.logging import get_logger

logger = get_logger(__name__)


class PromptTemplateManager:
    """Manager for prompt templates."""
    
    def __init__(
        self,
        system_prompt_file: str = "config/prompts/default_system_prompt.txt",
        custom_prompts_file: str = "config/prompts/custom_prompts.yml",
        template_engine: str = "jinja2"
    ):
        """
        Initialize prompt template manager.
        
        Args:
            system_prompt_file: Path to default system prompt
            custom_prompts_file: Path to custom prompts YAML
            template_engine: Template engine to use ('jinja2' or 'string')
        """
        self.system_prompt_file = Path(system_prompt_file)
        self.custom_prompts_file = Path(custom_prompts_file)
        self.template_engine = template_engine
        
        # Load custom prompts
        self.custom_prompts = self._load_custom_prompts()
        
        # Setup Jinja2 if needed
        if template_engine == "jinja2":
            prompts_dir = self.system_prompt_file.parent
            self.jinja_env = Environment(loader=FileSystemLoader(str(prompts_dir)))
        
        logger.info(f"Initialized PromptTemplateManager with engine={template_engine}")
    
    def load_system_prompt(self, prompt_name: Optional[str] = None) -> str:
        """
        Load system prompt.
        
        Args:
            prompt_name: Optional custom prompt name from YAML
            
        Returns:
            System prompt text
        """
        try:
            # Use custom prompt if specified
            if prompt_name and prompt_name in self.custom_prompts:
                prompt_config = self.custom_prompts[prompt_name]
                if 'system' in prompt_config:
                    logger.debug(f"Loaded custom prompt: {prompt_name}")
                    return prompt_config['system']
            
            # Load default system prompt
            if self.system_prompt_file.exists():
                with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                logger.debug("Loaded default system prompt")
                return prompt
            else:
                logger.warning(f"System prompt file not found: {self.system_prompt_file}")
                return self._get_fallback_system_prompt()
                
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            return self._get_fallback_system_prompt()
    
    def render_template(
        self,
        template_str: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Render a template with variables.
        
        Args:
            template_str: Template string
            variables: Variables for rendering
            
        Returns:
            Rendered template
        """
        try:
            if self.template_engine == "jinja2":
                template = Template(template_str)
                rendered = template.render(**variables)
            else:  # string formatting
                rendered = template_str.format(**variables)
            
            return rendered
            
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return template_str
    
    def get_user_template(self, prompt_name: Optional[str] = None) -> str:
        """
        Get user message template.
        
        Args:
            prompt_name: Optional custom prompt name
            
        Returns:
            User message template
        """
        if prompt_name and prompt_name in self.custom_prompts:
            prompt_config = self.custom_prompts[prompt_name]
            if 'user_template' in prompt_config:
                return prompt_config['user_template']
        
        # Default template
        return "Context: {{ context }}\n\nQuestion: {{ question }}"
    
    def _load_custom_prompts(self) -> Dict[str, Any]:
        """Load custom prompts from YAML."""
        try:
            if not self.custom_prompts_file.exists():
                logger.warning(f"Custom prompts file not found: {self.custom_prompts_file}")
                return {}
            
            with open(self.custom_prompts_file, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            
            if not prompts:
                logger.warning("Empty custom prompts file")
                return {}
            
            logger.info(f"Loaded {len(prompts)} custom prompt templates")
            return prompts
            
        except Exception as e:
            logger.error(f"Error loading custom prompts: {e}")
            return {}
    
    def _get_fallback_system_prompt(self) -> str:
        """Get fallback system prompt."""
        return """You are an AI assistant that answers questions based on provided source material. Follow these rules:

1. For each factual claim in your answer, include citation references [1], [2], etc.
2. Only use information from the provided context - do not add external knowledge
3. If you cannot find relevant information in the context, say so clearly
4. Be precise and accurate in your citations
5. When multiple sources support the same point, list all relevant citations
6. Provide comprehensive and well-structured answers

The context documents are numbered and you should reference them accordingly."""
