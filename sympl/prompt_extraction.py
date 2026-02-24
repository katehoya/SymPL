import re
from typing import List, Dict
from PIL import Image
from .utils import add_message, sympl_stage

def get_objects_of_interest(
    vlm_model,
    prompt_parser,
    image: Image.Image,
    prompt: str,
    num_tries: int = 10,
    conv_history: list = None,
    include_ref: bool = False,
    ref_viewer: str = None,
):

    match_found = False
    response_objs = None

    for try_idx in range(num_tries):
        if include_ref:
            prompt_objs = prompt_parser.get_prompt_by_type("get_objects_of_interest_ref")
            prompt_objs = prompt_objs.format(question=prompt, ref=ref_viewer)
            include_ref = False
        elif try_idx == 0:
            prompt_objs = prompt_parser.get_prompt_by_type("get_objects_of_interest")
            prompt_objs = prompt_objs.format(question=prompt)
        else:
            prompt_objs = prompt_parser.get_prompt_by_type("get_objects_of_interest_aux")
            prompt_objs = prompt_objs.format(question=prompt, response=response_objs)

        messages = add_message(
            [],
            role="user",
            text=prompt_objs,
        )

        response_objs = vlm_model.process_messages(messages)

        pattern_objs = prompt_parser.get_prompt_by_type("pattern_get_objects_of_interest")
        match_objs = re.findall(pattern_objs, response_objs)

        if len(match_objs) > 0:
            match_found = True
            break
    
    if not match_found:
        if conv_history is not None:
            conv_history += [
                {'text': prompt_objs, 'image': image.resize((400, 400))},
                {'text': response_objs, 'image': None},
            ]
        return None, conv_history

    match_objs = match_objs[-1]
    objs_of_interest = match_objs.strip().replace("[", "").replace("]", "").split(",")
    objs_of_interest = [obj.strip().lower().replace("'", "") for obj in objs_of_interest]

    if include_ref:
        if ref_viewer not in objs_of_interest:
            return None, conv_history

    if conv_history is not None:
        conv_history += [
            {'text': prompt_objs, 'image': image.resize((400, 400))},
            {'text': response_objs, 'image': None},
        ]
    
    return objs_of_interest, conv_history


def get_reference_viewer(
    vlm_model,
    prompt_parser,
    image: Image.Image,
    prompt: str,
    objects_of_interest: List[str],
    conv_history: list = None,
):

    prompt_ref_viewer = prompt_parser.get_prompt_by_type("get_reference_viewer")
    
    obj_str = ', '.join(objects_of_interest)
    prompt_ref_viewer = prompt_ref_viewer.format(question=prompt, obj_str=obj_str)

    messages = add_message(
        [],
        role="user",
        text=prompt_ref_viewer,
        image=None,
    )
    response_ref_viewer = vlm_model.process_messages(messages)

    pattern_ref_viewer = prompt_parser.get_prompt_by_type("pattern_reference_viewer")
    match_ref_viewer = re.findall(pattern_ref_viewer, response_ref_viewer)


    if len(match_ref_viewer) > 0:
        ref_viewer = match_ref_viewer[-1]
    else:
        ref_viewer = response_ref_viewer
    ref_viewer = ref_viewer.strip().lower().replace("'", "")

    if 'camera' in ref_viewer.lower():
        if ref_viewer.lower() not in objects_of_interest:
            objects_of_interest.append(ref_viewer)

    if ref_viewer.lower() not in objects_of_interest:
        objects_of_interest, _ = get_objects_of_interest(vlm_model, prompt_parser, image=image, prompt=prompt,conv_history=conv_history, include_ref = True, ref_viewer = ref_viewer)     

    if conv_history is not None:
        conv_history += [
            {'text': prompt_ref_viewer, 'image': None},
            {'text': response_ref_viewer, 'image': None},
        ]

    if objects_of_interest == None:
        return None, None, conv_history

    return ref_viewer, objects_of_interest, conv_history


@sympl_stage
def select_category_from_question(
    vlm_model,
    prompt_parser,
    prompt: str,
):
    original_prompt = prompt_parser.get_prompt_by_type("select_category")

    original_prompt = original_prompt.format(question=prompt)
    messages = add_message(
        [],
        role='user',
        text=original_prompt,
    )
    
    response_category = vlm_model.process_messages(messages)

    pattern_select_category = prompt_parser.get_prompt_by_type("pattern_select_category")
    match_category = re.findall(pattern_select_category, response_category)

    if len(match_category)==0:
        return None

    return match_category[0]
