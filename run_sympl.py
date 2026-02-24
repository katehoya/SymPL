import os
import sys
from PIL import Image
import yaml
from box import Box

# Ensure sympl is in path
sys.path.append("sympl/vision_modules")

from sympl.sympl_pipeline import SymPL

def run_minimal_sympl():
    # Load config
    config_path = "sympl/configs/qwenvl2_5_7b_instruct.yaml"
    image_path = "demo/image1.jpg"
    prompt = "Consider the real-world 3D locations and orientations of the objects. If I stand at the boy with long sleeve blue shirt's position facing where it is facing, is the boy in white on the left or right of me?"
    save_dir = "outputs/test"

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config)

    # Initialize SymPL (Assuming checkpoints are in place)
    sympl_pipeline = SymPL(config, device_vlm="cuda:0", device_vision="cuda:0")
    print("* [INFO] Loaded SymPL Pipeline!")

    # Load example image from demo
    image = Image.open(image_path).convert("RGB")
    print("* [INFO] Loaded image!")

    print(f"* [INFO] Running SymPL for prompt: {prompt}")
    
    os.makedirs(save_dir, exist_ok=True)

    response_sympl, conv_history = sympl_pipeline.run_sympl(
        image,
        prompt,
        trace_save_dir=save_dir,
        visualize_trace=True,
        return_conv_history=True
    )
    
    print(f"\n[RESPONSE]: {response_sympl}")
    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    run_minimal_sympl()
