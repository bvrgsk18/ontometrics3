from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference
from config import *
import os
os.environ["GOOGLE_API_KEY"] = GEMINI_API_TOKEN
context_precision = LLMContextPrecisionWithReference(llm=RAG_LLM_MODEL)

sample = SingleTurnSample(
    user_input="How many sales related call volume we have in west for may",
    reference="""In May 2025, the West region experienced a significant volume of sales-related calls. The data indicates that the Sales Related Call Volume for postpaid wireless services in the West was 2174.0. This volume of calls was associated with 85703.0 Wireless Gross Adds for postpaid services and resulted in 198980.0 Total Customer Interactions. The average handling time for these sales calls in the postpaid segment was 5.66 minutes.
In contrast, the prepaid wireless segment in the West also generated substantial sales call volume, totaling 3420.0 for May 2025. This was linked to 39230.0 Wireless Gross Adds for prepaid services, contributing to 121443.0 Total Customer Interactions. The average call handling time for prepaid sales calls was notably longer at 8.89 minutes.""",
    
    retrieved_contexts="""In May, the West region had a total of 5594.0 sales-related call volume. This was comprised of 2174.0 calls for postpaid services and 3420.0 calls for prepaid services."""
    )

context_precision.single_turn_ascore(sample)