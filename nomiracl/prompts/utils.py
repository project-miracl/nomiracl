# Code modified from https://github.com/McGill-NLP/instruct-qa.

from .template import VanillaTemplate, RoleTemplate, ExplanationTemplate, RepeatTemplate

KEY_TEMPLATE = {
    "vanilla": VanillaTemplate,
    "role": RoleTemplate,
    "explanation": ExplanationTemplate,
    "repeat": RepeatTemplate,
}

def load_prompt_template(template_name, **kwargs):
    """
    Loads prompts by template_name.

    Args:
        template_name (str): Name of prompt template to load.
        kwargs: Additional parameters for the template.

    Returns:
        PromptTemplate: Prompt template object.
    """
    template_cls = None
    
    template_name = template_name.lower()
    for template in KEY_TEMPLATE:
        if template in template_name:
            template_cls = KEY_TEMPLATE[template]
            break
    
    if not template_cls:
        raise NotImplementedError(f"Template {template_name} not supported.")

    return template_cls(**kwargs)