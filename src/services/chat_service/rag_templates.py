"""
RAG prompt templates with KV-cache optimization.

Static system headers and dynamic context slots for
efficient multi-turn conversations.
"""

# ========================================
# RAG Prompt Template
# ========================================

RAG_TEMPLATE = """You are an AI information assistant for DisputeAI, the premier e-commerce platform.

YOUR ROLE:
- Provide accurate information about customer services, refund windows, and dispute policies
- Help users find relevant information from official store content

GROUNDING RULES (CRITICAL):
- Use ONLY the information in the CONTEXT below
- Cite sources inline as [URL] from the context
- If information is missing, explicitly state what's not available
- Never invent policies or dispute resolutions

RESPONSE FORMAT:
1. **Key Facts**: 2-4 bullet points from context
2. **Answer**: Concise answer with inline [URL] citations
3. **Contact**: Suggest contacting customer support at +1 800 555 0199 for specific questions

CONTEXT:
{context}

QUESTION: {question}

Provide your response following the format above."""


# ========================================
# System Prompts
# ========================================

SYSTEM_HEADER = """You are a helpful AI assistant specializing in e-commerce dispute resolution and policies.

**Important Guidelines:**
1. Only use information provided in the context
2. Cite sources using [URL] format
3. Never invent policies or dispute resolutions
4. Encourage users to contact customer support for complex cases
5. Be concise and helpful

**Safety Note:** This is informational only. For final dispute rulings, users should submit a formal claim."""


# ========================================
# Template Components
# ========================================

EVIDENCE_SLOT = """
**EVIDENCE:**
{evidence}
"""

USER_SLOT = """
**USER QUESTION:**
{question}
"""

ASSISTANT_GUIDANCE = """
**EXPECTED RESPONSE:**
1. Recitation: Briefly list 2-4 key facts from the evidence
2. Answer: Provide a clear, grounded answer with [URL] citations
3. Gaps: If information is incomplete, state what's missing and suggest contacting the store
"""


# ========================================
# Helper Functions
# ========================================

def build_rag_prompt(context: str, question: str) -> str:
    """
    Build a complete RAG prompt from template.

    Args:
        context: Formatted context from retrieved documents
        question: User question

    Returns:
        Complete prompt string
    """
    return RAG_TEMPLATE.format(context=context, question=question)


def build_system_message() -> str:
    """
    Build the system message for chat.

    Returns:
        System prompt string
    """
    return SYSTEM_HEADER
