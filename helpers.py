from enum import Enum
import re
from typing import List, Tuple
class E2E_TOKEN(Enum):
    SEP_TOKEN = "<sep>"
    CONTEXT_TOKEN = "<context>"
    ANSWER_TOKEN = "<answer>"
    QUESTION_TOKEN = "<question>"
    PARAPHRASE_TOKEN = "<paraphrase>"
