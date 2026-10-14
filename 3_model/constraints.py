class BiasNegConstraint:
    """Clamp every bias to <= 0 after each optimizer step.

    The z+ rule ignores biases, so relevance is only conserved (sum of input relevance == output)
    when no bias can add positive contribution.
    """

    def __call__(self, module):
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias.data.clamp_(max=0.0)
