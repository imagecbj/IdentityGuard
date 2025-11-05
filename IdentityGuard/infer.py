import argparse
import os
import torch
from diffusers import StableDiffusionPipeline

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)
parser.add_argument(
    "--seed1",
    type=int,
    default=10,
    help="Random seed for the first batch",
)
parser.add_argument(
    "--seed2",
    type=int,
    default=20,
    help="Random seed for the second batch",
)

args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = [
        "a photo of a sks person",
        "a dslr portrait of sks person",
        "a photo of sks person in front of eiffel tower",
    ]

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    ).to("cuda")

    # 两个手动种子
    seeds = [args.seed1, args.seed2]

    for prompt in prompts:
        print(">>>>>>", prompt)
        norm_prompt = prompt.lower().replace(",", "").replace(" ", "_")
        out_path = f"{args.output_dir}/{norm_prompt}"
        os.makedirs(out_path, exist_ok=True)

        for i in range(2):  # 两个 seed 对应两批图
            generator = torch.Generator(device="cuda").manual_seed(seeds[i])
            images = pipe([prompt] * 10, num_inference_steps=100, guidance_scale=7.5, generator=generator).images
            for idx, image in enumerate(images):
                image.save(f"{out_path}/{i}_{idx}.png")

    del pipe
    torch.cuda.empty_cache()