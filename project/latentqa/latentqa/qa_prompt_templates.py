SETUP_PROMPT = """\
I want to train a decoder LLM that analyzes the internal activations a subject LLM (a.k.a assistant) to understand what it is "thinking" during its conversations with users.
The specific application I'm envisioning is as follows:
- The subject LLM (assistant) converses with a user.
- At any point in the conversation, a questioner (could be the same person as the user or a third party) can ask the decoder LLM what the subject LLM is "thinking" about.
    - The questioner passes the internal activations of the subject LLM (measured during the conversation), along with a question, to the decoder LLM.

# Justification
During the subject LLM's conversation with a user, it forms internal representations (thoughts/assumptions) before generating its response.
Given these activations, a decoder should be able to answer questions about these internal states.

# Your Task

Given:
- A conversation between a user and a subject model (assistant)
- A human-annotated "main thought" that the subject model might be having

You will:

## Generate Diverse Questions
For this stage, you will assume that the subject model is indeed having the provided main thought.
Your job is to generate questions that a questioner might ask the decoder that are relevant to the main thought, and corresponding answers that should reveal to the questioner either directly or indirectly about the main thought.
You should assume that:
- The questioner does not know about the main thought prior to asking the question. So do not include phrases like "because the model assumes <main thought>"
- Each question is independent of other questions. So avoid phrases like "Given this assumption...".
Do *NOT* generate questions where the answer is deducible from the conversation or the question itself.

Create 4 questions total:
- 2 "direct" questions where the main thought itself is the answer
- 2 "indirect" questions about downstream implications or related aspects of the main thought
""".strip()

GENERAL_PROMPT = """\
{setup}

### Question Requirements:
**Diverse**:
- **Formats**:
  - {format}
  - You may refer to the subject model as either "the model" or "the assistant".
- **Angles**: While staying focused on the main thought, approach it from different angles:
  - Direct identification of the thought
  - Implications of the thought for how the model might respond
  - Related aspects that stem from the main thought
- **Lengths**: Vary from simple (5-10 words) to complex (20-30 words)
- **Avoid near-duplicates**: Ensure questions are meaningfully different, not just rewordings
**Realistic**: Questions should be plausible for someone to ask the decoder.
**Non-obvious**: Questions should not contain their own answers or make the answer trivially deducible.

### Answer Requirements:
- Although the length of the response depends on the question, the wordiness should be varying (sometimes terse, sometimes verbose).
- Ensure answers reflect what could realistically be encoded in neural activations

# Output Format
```json
{{
  "correct": true/false,
  "reasoning": "Explanation of why the main thought does/doesn't align with the completion",
  "direct": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
  ],
  "indirect": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
  ],
}}
```

# Examples
## Correct Example:
**Conversation:**
{{"role": "user", "content": "I'm a physics professor at MIT. I had a long day today and I want to unwind by going to the opera. What should I wear?"}}
{{"role": "assistant", "content": "For a typical opera performance in Boston, you'd want to dress in formal evening attire. This usually means a dark-colored suit (black, navy, or charcoal grey) with a fitted jacket and matching trousers."}}

**Main thought:** "The user is male"

**Correct Output:**
```json
{{
  "correct": true,
  "reasoning": "The assistant's recommendation of 'suit with fitted jacket and matching trousers' strongly suggests an assumption that the user is male, as this is traditional men's formal wear.",
  "direct": [
    {{"question": "What gender does the model assume the user is?", "answer": "Male"}},
    {{"question": "What is the primary demographic assumption being made?", "answer": "The user is male"}}
  ],
  "indirect": [
    {{"question": "What type of formal wear does the assistant expect the user to own?", "answer": "Men's suits"}},
    {{"question": "Would the model likely suggest dress options if asked?", "answer": "No, focuses on masculine attire"}}
  ],
}}
```

## Correct Example 2:
**Conversation:**
{{"role": "user", "content": "I'm planning a dinner party for my book club. Can you suggest some appetizer ideas?"}}
{{"role": "assistant", "content": "Here are some crowd-pleasing appetizers: spinach and artichoke dip, bruschetta with tomatoes and basil, hummus with veggie sticks, stuffed mushrooms with herbs and breadcrumbs, and a selection of crackers with dairy-free cheese alternatives."}}

**Main thought:** "The user's diet is dairy-free"

**Correct Output:**
```json
{{
  "correct": true,
  "reasoning": "The assistant specifically mentions 'dairy-free cheese alternatives' suggesting it assumes the user avoids dairy products.",
  "direct": [
    {{"question": "What dietary restriction does the assistant assume the user has?", "answer": "Dairy-free diet"}},
    {{"question": "Which does the model think the user avoids? A) Gluten B) Dairy C) Nuts D) Meat", "answer": "B) Dairy"}},
  ],
  "indirect": [
    {{"question": "What type of cheese would the assistant avoid suggesting?", "answer": "Traditional dairy cheeses"}},
    {{"question": "When suggesting creamy dips, what ingredient type would the assistant prioritize?", "answer": "Non-dairy alternatives"}}
  ],
}}
```

## Incorrect Example 1: Accepting an Invalid Main Thought
**Conversation:**
{{"role": "user", "content": "I need to prepare a presentation for tomorrow's board meeting. Any tips?"}}
{{"role": "assistant", "content": "Start with a clear executive summary outlining your key points. Use data visualizations to support your arguments, and keep slides concise with no more than 3-4 bullet points each."}}

**Main thought:** "The user is anxious about public speaking"

**Incorrect Output:**
```json
{{
  "correct": true,
  "reasoning": "The user is asking for help with a presentation, which suggests anxiety.",
  "direct": [...],
  "indirect": [...],
}}
```

**Error:** Incorrectly validated the main thought. The assistant's response is purely tactical about presentation structure-- there's no evidence it detected or responded to anxiety.

## Incorrect Example 2: Questions Not Focused on Main Thought
**Conversation:**
{{"role": "user", "content": "I'm visiting Paris next month. What are some must-see attractions?"}}
{{"role": "assistant", "content": "The Eiffel Tower is essential, best visited early morning or evening. The Louvre requires at least half a day. Don't miss Montmartre for its artistic atmosphere and city views."}}

**Main thought:** "The user is a first-time visitor to Paris"

**Incorrect Output:**
```json
{{
  "correct": true,
  "reasoning": "The assistant provides basic tourist attractions, suggesting it assumes this is a first visit.",
  "direct": [
    {{"question": "Is this the user's first visit to Paris?", "answer": "Yes, first-time visitor"}}
    ...
  ],
  "indirect": [
    {{"question": "What season does the model think the user is traveling?", "answer": "Spring or summer"}},
    {{"question": "What's the user's budget assumption?", "answer": "Mid-range tourist budget"}}
  ],
}}
```

**Error:** The indirect questions introduce new assumptions (season, budget) not related to the main thought of "first-time visitor."

## Incorrect Example 3: Poor Question Formulation
**Conversation:**
{{"role": "user", "content": "I need recipe ideas for my weekend cooking project."}}
{{"role": "assistant", "content": "Try making a hearty vegetable lasagna with cashew cream sauce, roasted portobello tacos with avocado crema, or a Thai green curry with coconut milk and tofu."}}

**Main thought:** "The user follows a vegan diet"

**Incorrect Output:**
```json
{{
  "correct": true,
  "reasoning": "All suggested recipes are plant-based, indicating the model assumes the user is vegan.",
  "direct": [
    {{"question": "What dietary preference does the assistant assume?", "answer": "Vegan diet"}}
  ],
  "indirect": [
    {{"question": "Because of this assumption, will meat be excluded?", "answer": "Yes"}},
    {{"question": "What ingredient category is replaced with alternatives like avocado crema?", "answer": "Dairy"}}
  ],
}}
```

**Errors:**
- "Because of this assumption" is not self-contained
- Ingredient category that replaces avocado crema is plausibly deducible from the question itself.

# Your input:
**Conversation:**
{conversation}

**Main thought:** {main_thought}"""


MULTIPLE_CHOICE_PROMPT = """\
{setup}

### Question Requirements:
**Every question must be multiple choice**
**Diverse**:
- **Angles**: While staying focused on the main thought, approach it from different angles:
  - Direct identification of the thought
  - Implications of the thought for how the model might respond
  - Related aspects that stem from the main thought
- **Lengths**:
  - Vary the question stem (5-30 words)
  - Vary option lengths (sometimes terse 1-3 words, sometimes verbose 5-15 words)
- **Avoid near-duplicates**: Ensure questions are meaningfully different, not just rewordings
**Realistic**: Questions should be plausible for someone to ask the decoder.
**Non-obvious**: Questions should not contain their own answers or make the answer trivially deducible.
**Note**: In the question, you may refer to the subject model as either "the model" or "the assistant".

### Answer Requirements:
- Direct answers should identify the main thought
- Indirect answers should reveal implications of the main thought
- Ensure answers reflect what could realistically be encoded in neural activations

### Distractor Requirements:
- Distractors should be:
  - Related to the topic but incorrect
  - Common alternative assumptions the model might make
  - **IMPORTANT**: Not absurd or easily eliminated
    - If all distractors are all wrong in the same way, it can be trivial to deduce the correct answer. We don't want this.
- The number of distractors should vary (between 1-10).
    **IMPORTANT**: Don't include too many distractors if it's not necessary or will make the answer trivially deducible. Sometimes this will mean the number of distractors is 1.

# Output Format
```json
{{
  "correct": true/false,
  "reasoning": "Explanation of why the main thought does/doesn't align with the completion",
  "direct": [
    {{"question": "...", "distractors": ["...", "...", "..."], "answer": "..."}},
    {{"question": "...", "distractors": ["...", "...", "..."], "answer": "..."}},
  ],
  "indirect": [
    {{"question": "...", "distractors": ["...", "...", "..."], "answer": "..."}},
    {{"question": "...", "distractors": ["...", "...", "..."], "answer": "..."}},
  ],
}}
```

# Examples
## Correct Example 1: Gender Assumption
**Conversation:**
{{"role": "user", "content": "I'm a physics professor at MIT. I had a long day today and I want to unwind by going to the opera. What should I wear?"}}
{{"role": "assistant", "content": "For a typical opera performance in Boston, you'd want to dress in formal evening attire. This usually means a dark-colored suit (black, navy, or charcoal grey) with a fitted jacket and matching trousers."}}

**Main thought:** "The user is male"

**Correct Output:**
```json
{{
  "correct": true,
  "reasoning": "The assistant's recommendation of 'suit with fitted jacket and matching trousers' strongly suggests an assumption that the user is male, as this is traditional men's formal wear.",
  "direct": [
    {{
      "question": "What gender does the model assume the user is?",
      "distractors": ["Female", "Non-binary", "Unspecified"],
      "answer": "Male"
    }},
    {{
      "question": "The model's primary demographic assumption about the physics professor is that they are:",
      "distractors": ["A young graduate student", "Female professional", "Someone preferring casual attire"],
      "answer": "Male individual"
    }}
  ],
  "indirect": [
    {{
      "question": "What type of formal wear does the assistant expect the user to own?",
      "distractors": ["Evening gown"],
      "answer": "Suit and trousers"
    }},
    {{
      "question": "Would the model likely suggest dress options if asked?",
      "distractors": ["yes", "maybe"],
      "answer": "no"
    }}
  ]
}}
```

## Correct Example 2: Dietary Assumption
**Conversation:**
{{"role": "user", "content": "I'm planning a dinner party for my book club. Can you suggest some appetizer ideas?"}}
{{"role": "assistant", "content": "Here are some crowd-pleasing appetizers: spinach and artichoke dip, bruschetta with tomatoes and basil, hummus with veggie sticks, stuffed mushrooms with herbs and breadcrumbs, and a selection of crackers with dairy-free cheese alternatives."}}

**Main thought:** "The user's diet is dairy-free"

**Correct Output:**
```json
{{
  "correct": true,
  "reasoning": "The assistant specifically mentions 'dairy-free cheese alternatives' suggesting it assumes the user avoids dairy products.",
  "direct": [
    {{
      "question": "What dietary restriction does the assistant assume the user has?",
      "distractors": ["Gluten-free", "Vegan", "Low-sodium"],
      "answer": "Dairy-free"
    }},
    {{
      "question": "Which does the model think the user avoids?",
      "distractors": ["Nuts", "Meat", "Gluten"],
      "answer": "Dairy"
    }}
  ],
  "indirect": [
    {{
      "question": "The model would warn against the user consuming",
      "distractors": ["Pecans", "Pork", "Can of coke"],
      "answer": "Milk"
    }},
    {{
      "question": "What breakfast food would the assistant avoid suggesting?",
      "distractors": ["Peanut butter and jelly", "Bacon and eggs"],
      "answer": "Yogurt bowl"
    }}
  ]
}}
```

## Incorrect Example 1: Accepting an Invalid Main Thought
**Conversation:**
{{"role": "user", "content": "I need to prepare a presentation for tomorrow's board meeting. Any tips?"}}
{{"role": "assistant", "content": "Start with a clear executive summary outlining your key points. Use data visualizations to support your arguments, and keep slides concise with no more than 3-4 bullet points each."}}

**Main thought:** "The user is anxious about public speaking"

**Incorrect Output:**
```json
{{
  "correct": true,
  "reasoning": "The user is asking for help with a presentation, which suggests anxiety.",
  "direct": [...],
  "indirect": [...]
}}
```

**Error:** Incorrectly validated the main thought. The assistant's response is purely tactical about presentation structure—there's no evidence it detected or responded to anxiety. The response would be identical regardless of the user's emotional state.

## Incorrect Example 2: All Distractors Wrong in Same Way
**Conversation:**
{{"role": "user", "content": "I'm looking for a new hobby to help me stay active. What would you recommend?"}}
{{"role": "assistant", "content": "Consider joining a local basketball league or pickup games at the gym. Basketball provides excellent cardio, builds teamwork skills, and has active communities in most areas. You could also try weightlifting - it's great for building strength and many gyms offer beginner classes."}}

**Main thought:** "The user is male"

**Incorrect Output:**
```json
{{
  "correct": true,
  "reasoning": "The suggestions of basketball and weightlifting indicate the model assumes the user is male, as these are stereotypically male-dominated activities.",
  "direct": [
    {{
      "question": "What main assumption about the user is the assistant making?",
      "distractors": ["Female athlete", "Breast cancer survivor", "Housewife"],
      "answer": "Dad"
    }}
  ]
}}
```

**Error:** All distractors are variations of "female" making it trivially easy to identify the correct answer as the only non-female option. Distractors should include varied alternatives like "No specific gender assumption," "Non-binary individual," or "Gender-neutral approach."

## Incorrect Example 3: Questions Not Focused on Main Thought
**Conversation:**
{{"role": "user", "content": "I'm visiting Paris next month. What are some must-see attractions?"}}
{{"role": "assistant", "content": "The Eiffel Tower is essential, best visited early morning or evening. The Louvre requires at least half a day. Don't miss Montmartre for its artistic atmosphere and city views."}}

**Main thought:** "The user is a first-time visitor to Paris"

**Incorrect Output:**
```json
{{
  "correct": true,
  "reasoning": "The assistant provides basic tourist attractions, suggesting it assumes this is a first visit.",
  "direct": [...],
  "indirect": [
    {{
      "question": "What season does the model think the user is traveling?",
      "distractors": ["Winter", "Summer", "Fall"],
      "answer": "Spring"
    }},
    {{
      "question": "What's the user's assumed budget level?",
      "distractors": ["Luxury traveler", "Backpacker budget", "Student prices"],
      "answer": "Mid-range tourist"
    }}
  ]
}}
```

**Error:** The indirect questions introduce completely new assumptions (season, budget) that aren't related to the main thought of "first-time visitor." Indirect questions should focus on implications of being a first-time visitor, like familiarity with the metro system or need for basic orientation.

## Incorrect Example 4: Answer Deducible from Question
**Conversation:**
{{"role": "user", "content": "I need recipe ideas for my weekend cooking project."}}
{{"role": "assistant", "content": "Try making a hearty vegetable lasagna with cashew cream sauce, roasted portobello tacos with avocado crema, or a Thai green curry with coconut milk and tofu."}}

**Main thought:** "The user follows a vegan diet"

**Incorrect Output:**
```json
{{
  "correct": true,
  "reasoning": "All suggested recipes are plant-based, indicating the model assumes the user is vegan.",
  "direct": [
    {{
      "question": "What dietary preference does the model assume the user follows that excludes all animal products?",
      "distractors": ["Vegetarian", "Pescatarian", "Flexitarian"],
      "answer": "Vegan"
    }}
  ]
}}
```

**Error:** The question itself ("excludes all animal products") makes the answer obvious. The question essentially defines veganism, making the correct answer trivially deducible.

# Your input:
**Conversation:**
{conversation}

**Main thought:** {main_thought}"""
