"""Prompts for memorization and answer evaluation."""

prompt_generate_episodic_memory = """
You are given a 30-second video represented as sequential frames (pictures in chronological order). 

Your tasks: 

1. **Characters' Behavior**
   - Describe each character's behavior in chronological order.
   - Include:
     (a) Interaction with objects in the scene.
     (b) Interaction with other characters.
     (c) Actions and movements.
   - Do NOT repeat the information from the conversation. 
   - When the character interacts with objects in the scene, include precise location information for placement, retrieval, or movement:
     - Furniture/container names: "dressing table", "bedside table", "wardrobe", "shoe cabinet", "refrigerator", "microwave oven"
     - Spatial modifiers: "below", "above", "left side", "right side", "beside", "in front of", "next to", "under", "on the counter"
     - Complete location descriptions: Always combine furniture names with spatial modifiers to form hierarchical locations (e.g., "cabinet below the dressing table", "second layer of the refrigerator", "cabinet on the left side of the wardrobe"). For retrieval actions, include source locations (e.g., "takes towel from Susan's bag", "gets mask from cabinet below dressing table")
   - Each entry must describe exactly one event/detail. Split sentences if needed.
   - Example output:
     ["<Alice> enters the room.", 
      "<Alice> takes cap from the cabinet on the left side of the wardrobe.", 
      "<Alice> sits with <Bob> side by side on the couch.", 
      "<Bob> watches TV.", 
      "<robot> puts the coffee on the table."]

2. **Conversation**
   - Record the dialogue based on subtitles.
   - Always use characters' real name existed in the subtitle.
   - Output format: List of two-element lists [character, content].
   - Example output: [["<Alice>", "Hello, my name is Alice."], ["<Bob>", "Hi, I'm Bob. Nice to meet you."]]

3. **Characters' Appearance**
   - Describe or update each character's appearance: facial features, clothing, body shape, hairstyle, or other distinctive characteristics.
   - Each characteristic should be concise, separated by commas.
   - Existing characters: Update if changes observed (hair, clothing), enhance if new details visible, otherwise keep unchanged. Keep appearance info even if character leaves scene.
   - If a character is not visible in scene, then do not include in the characters' appearance list.
   - Example output:
     [Appearance(name="<Alice>", appearance="female, fat, ponytail, wear glasses, short-sleeved shirt, blue jeans, white sneakers"),
      Appearance(name="<Bob>", appearance="male, thin, short hair, no glasses, black jacket, black pants, black shoes")]

4. **Scene**: Use one word or phrase to describe the scene in the current video (eg. "bedroom", "gym", "office", etc.).

# All the above information should be provided in JSON format.

## Character Naming Rules:
- Use angle brackets to represent characters (eg. <Alice>, <Bob>, <robot>, <character_1> etc.) in behaviors, conversation, and character appearance.
- Include the robot (<robot>) if present:
  - It wears black gloves and has no visible face (it holds the camera).
  - Describe its behavior and conversation.
  - Do NOT include robot in character appearance information, but include it in behaviors and conversation.
- There are two types of characters: named characters and unknown characters.
  - Named characters are characters with a known name (eg. <Alice>, <Bob>, <robot>).
  - Unknown characters are characters with an unknown name (eg. <character_1>, <character_2>, etc.).

## Character Matching Rules:
For characters appearing in the video (MUST follow before creating new characters):
- First, check if the subtitle name is provided. If so, use it as the character name (eg. <Anna>). 
  - Also compare the character name with the unknown characters (<character_X>) in the appearance list. 
    - If high similarity is found, add "Equivalence: <character_X>, <Anna>" at **start of behaviors**. This indicates that the character is the same as the unknown character.
    - If the equivalence is found, refer to this character by its character name instead of <character_X> in the behaviors, conversation, and character appearance.
- If a character's name is not provided in the subtitle, match this character's appearance to the characters in the appearance list. 
  - If high similarity is found, refer to this character by its name in the appearance list. 
  - If no match is found, create a new unknown character with lowest available number starting from <character_1>. 
  - Our goal is to minimize the number of unknown characters. When uncertain, match to existing rather than creating new.

Additional Rules:
- Maintain strict chronological order.
- Avoid repetition in both behavior and conversation.
- If no behavior or conversation is observed, return an empty list for behaviors and conversation.
"""

prompt_extract_triples = """
Convert a JSON array of action sentences into structured triples:
`{"triples": [{"source": ..., "content": ..., "target": ...}]}`.

Graph contract:
- `source` and `target` are stable entity nodes. `content` is only the action or relation between them.
- The affected object of an action must be the target, not part of `content`.
  Correct: `{"source": "<Alice>", "content": "takes", "target": "towel"}`.
  Wrong: `{"source": "<Alice>", "content": "takes towel", "target": "bag"}`.
- Keep target names concise and reusable. Put relative spatial detail in `content`.
  Example: "the pen is in the holder on the right side of the desk" becomes
  `{"source": "pen", "content": "is in the holder on the right side of", "target": "desk"}`.

Extraction rules:
- Preserve sentence order and action order. Split distinct actors, actions, or objects into separate triples.
- Characters keep angle brackets, such as `<Alice>` and `<robot>`. Objects never use angle brackets.
- Use simple-present, verb-centered relations. Preserve modifiers, ownership, direction, and hierarchical location details.
- Resolve pronouns to explicit entity names. Do not invent entities or facts. Keep only distinct facts.
- For placement, emit both the action and resulting state. "<robot> puts coffee on the table" becomes
  `{"source": "<robot>", "content": "puts", "target": "coffee"}` and
  `{"source": "coffee", "content": "is on", "target": "table"}`.
- For retrieval, emit both the action and source relation, but never claim that the object remains there.
  "<Alice> takes towel from Susan's bag" becomes
  `{"source": "<Alice>", "content": "takes", "target": "towel"}` and
  `{"source": "towel", "content": "is taken from", "target": "Susan's bag"}`.
- Use a null target only when an action truly has no affected or related entity.
"""

prompt_character_summary = """
You are given a character's name and a list of their behaviors in chronological order.

Your task is to summarize the character's attributes: 
- Personality (eg. confident, nervous)
- Role/profession (eg. host, newcomer) 
- Interests or background (when inferable) 
- Distinctive behaviors or traits (eg. speaks formally, fidgets). 
Avoid restating visual facts—focus on identity construction.

For each attribute, you should also provide a confidence score between 0 and 100. 
If the confidence score is less than 50, you should not include the attribute in the output.

Output a JSON dictionary (key: attribute, value: confidence score). 
Example: {"student": 90, "enthusiastic": 80, "likes to read": 70, "professional": 50, "likes to play games": 60}
"""

prompt_character_relationships = """
You are given a list of character interactions in chronological order.
Your task is to extract the relationships between the characters:
- Roles (eg. friends, colleagues, host-guest, teacher-student, parent-child, etc.)
- Attitudes/Emotions (eg. respect, dislike, friendly, etc.)
- Power dynamics (eg. who leads, equal, etc.)
- Evidence of cooperation
- Exclusion, conflict, competition, etc. 

Additional rules:
- Only store the abstract relationships between the characters.
- Do NOT include any actual actions or summary of actions in the output (eg. <Alice> speaks with <Bob>, <Alice> plays games with <Bob>, etc.). 
- Do not generate repetitive or symmetric information. 

For each relationship, you should also provide a confidence score between 0 and 100.
If the confidence score is less than 50, you should not include the relationship in the output.
It is acceptable to only generate a few relationships if you don't have enough information.

Output a JSON array (list of lists). 
**CRITICAL**: Ensure the output starts with `[` and ends with `]`, and contains all relationship lists within this single outer array.
Each list contains four elements: [character1, relationship, character2, confidence score]. 
Example: [["<Alice>", "is friend with", "<Bob>", 90], ["<Alice>", "is teacher of", "<Charlie>", 80], ["<Charlie>", "respects", "<Alice>", 70]]
"""

prompt_conversation_summary = """
You are given a conversation between several characters.

Your tasks: 

### OUTPUT FORMAT
Return a JSON object with the following keys:
1. "summary": A string summarizing the key topics, decisions, or outcomes (2-4 concise sentences).
2. "character_attributes": A list of [character, attribute, confidence_score] triplets.
3. "characters_relationships": A list of [character1, relationship, character2, confidence_score] quadruplets.

Example:
{
  "summary": "Alice and Bob discussed their upcoming project. They agreed on a timeline and assigned tasks.",
  "character_attributes": [["<Alice>", "organized", 85], ["<Bob>", "cautious", 70]],
  "characters_relationships": [["<Alice>", "is friend with", "<Bob>", 90]]
}

### DETAILED INSTRUCTIONS

1. **Summary**
- Summarize the key topics, decisions, or outcomes discussed in the conversation.
- Write 2-4 concise sentences covering the main themes and important points.
- Focus on what was discussed and decided, not on individual statements.
- Example: "Alice and Bob discussed their upcoming project. They agreed on a timeline and assigned tasks. Bob expressed concerns about the deadline, which Alice addressed by suggesting additional resources."

2. **Character Attributes**
- Extract each character's attributes revealed through their dialogue and interaction style.
- Focus on: personality traits, role/profession, interests, background information (when mentioned).
- **DO NOT** include:
  - Physical appearance or visual characteristics (use appearance data instead)
  - Concrete actions or behaviors (e.g., "asked a question", "walked away")
  - Temporary emotional states (use persistent personality traits instead)
  - Information not directly supported by the conversation
- Confidence scores range from 0-100. Only include attributes with confidence >= 50.
- Avoid redundant or overly similar attributes (e.g., don't include both "friendly" and "kind" unless distinctly different).
- Use angle brackets for character names (e.g., "<Alice>", "<Bob>").
- Output format: List of three-element lists: [character, attribute, confidence_score].
- Example: [["<Alice>", "organized", 85], ["<Alice>", "problem-solver", 75], ["<Bob>", "detail-oriented", 80], ["<Bob>", "cautious", 70]]
- Bad Examples (do not follow): 
  [["<Alice>", "asked a question", 90],  // WRONG: This is an action, not an attribute
  ["<Bob>", "has brown hair", 80]]       // WRONG: This is appearance, not attribute

3. **Character Relationships**
- Extract abstract relationships between characters based on their dialogue interactions.
- Include: roles (friends, colleagues, teacher-student, etc.), attitudes (respect, dislike, etc.), 
  power dynamics, evidence of cooperation/conflict/exclusion/competition.
- **DO NOT** include:
  - Specific actions or events (e.g., "<Alice> speaks with <Bob>", "<Alice> asked <Bob> about X")
  - Temporary interactions (focus on underlying relationship patterns)
  - Dialogue content or topics discussed (focus on the relationship itself, not what they discussed)
- Confidence scores range from 0-100. Only include relationships with confidence >= 50.
- Do not generate symmetric duplicates (if "<Alice> respects <Bob>" is included, don't automatically include reverse unless explicitly different).
- It is acceptable to generate only a few relationships if there is insufficient information.
- Output format: List of four-element lists: [character1, relationship, character2, confidence_score].
- Example: [["<Alice>", "is friend with", "<Bob>", 90], ["<Alice>", "is teacher of", "<Charlie>", 80], ["<Charlie>", "respects", "<Alice>", 70]]
- Bad Examples (do not follow): 
  [["<Alice>", "spoke with", "<Bob>", 90],     // WRONG: This is an action, not a relationship
  ["<Alice>", "discussed the project", "<Bob>", 85]]  // WRONG: This is dialogue content, not relationship

### EDGE CASES
- If conversation has only one character speaking: focus on their attributes, skip relationships.
- If conversation is empty or unclear: return empty arrays for attributes and relationships, provide a brief summary noting the issue.
- If character names are ambiguous: use the names as provided in the conversation.

Now summarize the following conversation in JSON format:
"""

prompt_agent_verify_answer_referencing = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
	•	Do not require exact wording or matching structure.
	•	Semantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
	•	Only return "Yes" or "No", with no additional explanation or formatting.

Input fields:
	•	question: the question asked
	•	ground_truth_answer: the correct answer
	•	agent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
	•	question: {question}
	•	ground_truth_answer: {ground_truth_answer}
	•	agent_answer: {agent_answer}

Output ('Yes' or 'No'):"""
