# Code modified from https://github.com/McGill-NLP/instruct-qa. All credits goes to them!

from .template import HAGRIDTemplate, RerankTemplate, InstructQATemplate, YESNOTemplate, FewShotPointwiseTemplate, FewShotRelevanceTemplate, OrcaDPOTranslationTemplate


def load_prompt_template(template_name, **kwargs):
    """
    Loads model by name.

    Args:
        template_name (str): Name of prompt template to load.
        kwargs: Additional parameters for the generator (e.g., temperature).

    Returns:
        BaseGenerator: Generator object.
    """
    if "hagrid" in template_name:
        template_cls = HAGRIDTemplate
    elif "rerank" in template_name:
        template_cls = RerankTemplate
    elif "instructqa" in template_name:
        template_cls = InstructQATemplate
    elif "yesno" in template_name:
        template_cls = YESNOTemplate
    elif "fewshot" in template_name or "pointwise" in template_name:
        template_cls = FewShotPointwiseTemplate
    elif "relevance" in template_name:
        template_cls = FewShotRelevanceTemplate
    elif "orca" in template_name:
        template_cls = OrcaDPOTranslationTemplate
    else:
        raise NotImplementedError(f"Template {template_name} not supported.")

    return template_cls(**kwargs)