# ---------------------------------------------------------------- # 
# Predefined prompt templates
# ---------------------------------------------------------------- # 

# Main prompt for 'get_objects_of_interest'
PROMPT_GET_OBJECTS_OF_INTEREST = """
### Situation Description
Given a spatial reasoning question, please return all the words the represent the entities that are included in the question.

# Example 1
[Question] You are standing at the airplane's position, facing where it is facing. Is the the person on your left or right?
[Detect] [airplane, person]

# Examples 2
[Question] From the old man's perspective, is the person wearing a hat on the left of the green car?
[Detect] [old man, person wearing a hat, green car]

# Examples 3
[Question] From the car's perspective, which is on the right side: the person or the tree?
[Detect] [car, person, tree]

# Example 4
[Question] Where is the bus in relation to the horse? Answer with front and behind.
[Detect] [bus, horse]

### Your Task
Now, given the question below, please identify the entities that are included in the question.
All the results return as a format [Detect] [object_1, object_2, ...].

[Question] {question}
[Detect]
"""

# Main prompt for 'get_objects_of_interest'
PROMPT_GET_OBJECTS_OF_INTEREST_REF = """
### Situation Description
Given a spatial reasoning question, please return all the words the represent the entities that are included in the question include the word {ref}.

# Example 1
[Question] You are standing at the airplane's position, facing where it is facing. Is the the person on your left or right?
[Detect] [airplane, person]

# Examples 2
[Question] From the old man's perspective, is the person wearing a hat on the left of the green car?
[Detect] [old man, person wearing a hat, green car]

# Examples 3
[Question] From the car's perspective, which is on the right side: the person or the tree?
[Detect] [car, person, tree]

# Example 4
[Question] Where is the bus in relation to the horse? Answer with front and behind.
[Detect] [bus, horse]

### Your Task
Now, given the question below, please identify the entities that are included in the question.
All the results return as a format [Detect] [object_1, object_2, ...].

[Question] {question}
[Detect]
"""


# Auxiliary prompt for when the VLM fails to match the format for 'get_objects_of_interest'
PROMPT_GET_OBJECTS_OF_INTEREST_AUX = """
Looks like your response is not in the correct format!

Previous response: {response}

Please modify your response to the correct format.

[Question] {question}
[Detect]
"""


# Pattern to match for 'get_objects_of_interest'
PATTERN_GET_OBJECTS_OF_INTEREST = r"\[[^\]]*\]"

# Prompt for 'get_reference_viewer'
PROMPT_GET_REFERENCE_VIEWER = """
Given a question about spatial reasoning, we want to extract the **perspective** of the question.

If the question is from the camera's perspective or cannot mention the perspective, return ++camera++.
Never return anything else.

### Example 1
[Question] From the camera's perspective, which side of the car is facing the camera?
[Perspective] ++camera++

### Example 2
[Question] Where is the bus in relation to the horse? Answer with front and behind.
[Perspective] ++camera++

### Example 3
[Question] From the woman with blue bag's perspective, is the tree on the left or right?
[Perspective] ++woman with blue bag++

### Example 4
[Question] If you are standing at the airplane's position, facing where it is facing, is the car on your left or right?
[Perspective] ++airplane++

### Example 5
[Question] Where is the teddy bear in relation to the dining table? Answer with front and behind.
[Perspective] ++camera++

### Example 6
[Question] Which object is the dog facing towards, the woods or the frisbee?
[Perspective] ++dog++

### Example 7
[Question] If I stand at the shepherd's position facing where it is facing, is the sheep visible or not
[Perspective] ++shepherd++


### Your Task
Given the question below, please specify the **perspective** from which the question is asked.
After "[Perspective]" at the end of this prompt, you must return the answer for the base object in the "object_name" field, following the format : ++object_name++

“object_name” must be selected only from the [Option] list provided below.
Never return any answer outside of these options.
Just include ++ in front of and behind of the selected "object_name" candidate. Never change anything else.

[Question] {question}
[Options] {obj_str}, camera

[Perspective]
"""

# Pattern to match for 'get_reference_viewer'
PATTERN_GET_REFERENCE_VIEWER = r"\+\+(.*?)\+\+"

# Prompt for 'convert_to_ego' - remove perspective description

PROMPT_LEFT_RIGHT_SIMPLE_ONE = """
This is an image of a simple 2D Scene.

# Task
Based on the image, please answer the following question.
[Question] In the image, is the {obj} dot located in the 'yellow' area or the 'black' area?

Please only return the answer.
"""

PROMPT_LEFT_RIGHT_SIMPLE_TWO = """
This is an image of a simple 2D Scene.

# Task
Based on the image, please answer the following question.
[Question] In the image, which dot is located in the 'yellow' area, the {obj_1} dot or the {obj_2} dot?

Please only return the answer.
"""

PROMPT_CLOSER_SIMPLE = """
This is an image of a simple 2D Scene.

# Task
Based on the image, please answer the following question.
[Question] In the image, which dot is located in the 'yellow' area, the {obj_1} dot or the {obj_2} dot?

Please only return the answer.
"""

PROMPT_VISIBILITY_SIMPLE = """
This is an image of a simple 2D Scene.

# Task
Based on the image, please answer the following question.
[Question] In the image, is the {obj} dot located in the 'yellow' area or the 'black' area?

Please only return the answer.
"""

PROMPT_FACING_SIMPLE = """
This is an image of a simple 2D Scene.

# Task
Based on the image, please answer the following question.
[Question] In the image, which dot is located in the 'yellow' area, the {obj_1} dot or the {obj_2} dot?

Please only return the answer.
"""

PROMPT_FRONT_BEHIND_SIMPLE = """
This is an image of a simple 2D Scene.

# Task
Based on the image, please answer the following question.
[Question] In the image, which dot is located in the 'yellow' area, the {obj_1} dot or the {obj_2} dot?

Please only return the answer.
"""

PROMPT_ABOVE_BELOW_SIMPLE = """
This is an image of a simple 2D Scene.

# Task
Based on the image, please answer the following question.
[Question] In the image, which dot is located in the 'yellow' area, the {obj_1} dot or the {obj_2} dot?

Please only return the answer.
"""

# Prompt for 'select_category'
PROMPT_SELECT_CATEGORY = """
Given a question about spatial reasoning, we want to extract the category of the question.
The words inside ** ** in the [Question] are the key elements of that [Category].
Depending on the expression, words such as "visible" or "facing" may appear in [Question]. 
However, the mere presence of these words does not determine that [Category] should be "visibility" or "facing."
Refer to the parts highlighted with ** ** in the examples and select the most appropriate [Category].

### Example 1
[Question] If I stand at the man in cowboy hat's position facing where it is facing, is the bus stop **on the left or right** of me?
[Category] --left_right--

### Example 2
[Question] Which object is the small boat **facing towards**, the cruise ship or the Statue of Liberty?
[Category] --facing--

### Example 3
[Question] From the refrigerator's perspective, **which object is located closer** to the viewer, the camel or the cat?
[Category] --closer--

### Example 4
[Question] From the duck's perspective, **which object between** chair, refrigerator **is visible**?
[Category] --facing--

### Example 5
[Question] From the horse's perspective, **which object is located on the left side**, the camel or the duck?
[Category] --left_right--

### Example 6
[Question] If I stand at the dog's position facing where it is facing, is the white board **visible or not**
[Category] --visibility--

### Example 7
[Question] From the car's perspective, **which object is located above**, the person or the traffic sign?
[Category] --above_below--

### Example 8
[Question] Where is the person in the relation to the dog? Answer with left and right.
[Category] --left_right--

### Example 9
[Question] Where is the potted plant in the relation to the cat? Answer with front and behind.
[Category] --front_behind--

### Your Task
Given the question below, please specify the category from which the question is asked.
You must return in the format: [Category] --category_name--
"object_name" is selected from [Options] below. 
Never return a response that is not included in the given options.
Never change the format and capitalization from the option when returns response.

[Question] {question}
[Options] visibility / left_right / facing / closer / above_below / front_behind

[Category]
"""
# Pattern to match for 'get_reference_viewer'
PATTERN_GET_SELECTED_CATEGORY = r"\-\-(.*?)\-\-"

# ---------------------------------------------------------------- # 

class PromptParser:
    def __init__(self, config):
        self.config = config
        # Define predefined prompts for different use cases
        self.predefined_prompts = {
            "get_objects_of_interest": PROMPT_GET_OBJECTS_OF_INTEREST,
            "get_objects_of_interest_aux": PROMPT_GET_OBJECTS_OF_INTEREST_AUX,
            "get_objects_of_interest_ref": PROMPT_GET_OBJECTS_OF_INTEREST_REF,
            "pattern_get_objects_of_interest": PATTERN_GET_OBJECTS_OF_INTEREST,
            "get_reference_viewer": PROMPT_GET_REFERENCE_VIEWER,
            "pattern_reference_viewer": PATTERN_GET_REFERENCE_VIEWER,
            "lr_simple_one" : PROMPT_LEFT_RIGHT_SIMPLE_ONE,
            "lr_simple_two" : PROMPT_LEFT_RIGHT_SIMPLE_TWO,
            "closer_simple" : PROMPT_CLOSER_SIMPLE,
            "visibility_simple" : PROMPT_VISIBILITY_SIMPLE,
            "facing_simple" : PROMPT_FACING_SIMPLE,
            "select_category" : PROMPT_SELECT_CATEGORY,
            "fb_simple" : PROMPT_FRONT_BEHIND_SIMPLE,
            "ab_simple" : PROMPT_ABOVE_BELOW_SIMPLE,
            "pattern_select_category" : PATTERN_GET_SELECTED_CATEGORY,
        }

    def get_prompt_by_type(self, prompt_type: str) -> str:
        if prompt_type not in self.predefined_prompts:
            raise KeyError(f"Unknown prompt type: {prompt_type}. Available types: {list(self.predefined_prompts.keys())}")
        
        return self.predefined_prompts[prompt_type]

    def list_available_prompts(self) -> list:

        return list(self.predefined_prompts.keys())

