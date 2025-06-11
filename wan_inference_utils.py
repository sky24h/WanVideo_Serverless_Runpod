import os
import torch
import imageio
import numpy as np
import tempfile
import sys
sys.path.append("ComfyUI")

from custom_nodes.WanVideoWrapper.nodes import (
    WanVideoBlockSwap,
    WanVideoLoraSelect,
    WanVideoModelLoader,
    LoadWanVideoT5TextEncoder,
    WanVideoTextEncode,
    WanVideoEmptyEmbeds,
    WanVideoSLG,
    WanVideoSampler,
    WanVideoVAELoader,
    WanVideoDecode,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_video(frames: torch.Tensor, seed: int) -> str:
    output_video_path = tempfile.NamedTemporaryFile(prefix="{}_".format(seed), suffix=".mp4").name
    frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)
    writer = imageio.get_writer(output_video_path, fps=24, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=1)
    for image in frames_np:
        writer.append_data(image)
    writer.close()
    return output_video_path


def check_data_format(job_input: dict) -> dict:
    # must have prompt in the input, otherwise raise error to the user
    if "prompt" in job_input:
        prompt = job_input["prompt"]
    else:
        raise ValueError("The input must contain a prompt.")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")

    # optional params, make sure they are in the right format here, otherwise raise error to the user
    steps = job_input["steps"] if "steps" in job_input else None
    num_frames = job_input["num_frames"] if "num_frames" in job_input else None
    width = job_input["width"] if "width" in job_input else None
    height = job_input["height"] if "height" in job_input else None
    n_prompt = job_input["n_prompt"] if "n_prompt" in job_input else None
    cfg = job_input["cfg"] if "cfg" in job_input else None
    shift = job_input["shift"] if "shift" in job_input else None
    seed = job_input["seed"] if "seed" in job_input else None

    # check optional params
    if steps is not None and not isinstance(steps, int):
        raise ValueError("steps must be an integer.")
    if num_frames is not None and not isinstance(num_frames, int):
        raise ValueError("num_frames must be an integer.")
    if width is not None and not isinstance(width, int):
        raise ValueError("width must be an integer.")
    if height is not None and not isinstance(height, int):
        raise ValueError("height must be an integer.")
    if n_prompt is not None and not isinstance(n_prompt, str):
        raise ValueError("n_prompt must be a string.")
    if cfg is not None and not isinstance(cfg, float) and not isinstance(cfg, int):
        raise ValueError("cfg must be a float or an integer.")
    if shift is not None and not isinstance(shift, float) and not isinstance(shift, int):
        raise ValueError("shift must be a float or an integer.")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    return {
        "prompt": prompt,
        "steps": steps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "n_prompt": n_prompt,
        "cfg": cfg,
        "shift": shift,
        "seed": seed,
    }


class WanVideo:
    def __init__(self, lora_name=None, transformer_name=None, t5_model_name=None, vae_name=None):
        if lora_name is None:
            lora_name = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
        if transformer_name is None:
            transformer_name = "Wan2_1-T2V-14B_fp8_e5m2.safetensors"
        if t5_model_name is None:
            t5_model_name = "umt5-xxl-enc-bf16.safetensors"
        if vae_name is None:
            vae_name = "Wan2_1_VAE_bf16.safetensors"

        with torch.no_grad():
            block_swap = WanVideoBlockSwap()
            block_swap = block_swap.setargs(
                blocks_to_swap=38, offload_img_emb=False, offload_txt_emb=False, use_non_blocking=True, vace_blocks_to_swap=13
            )[0]

            lora_select = WanVideoLoraSelect()
            lora_select = lora_select.getlorapath(lora=lora_name, strength=0.5)[0]

            wanvideo = WanVideoModelLoader()
            self.wanvideo = wanvideo.loadmodel(
                transformer_name,
                base_precision="fp16",
                load_device="cpu",
                quantization="disabled",
                block_swap_args=block_swap,
                lora=lora_select,
            )[0]

            text_encoder = LoadWanVideoT5TextEncoder()
            self.text_encoder = text_encoder.loadmodel(t5_model_name, precision="bf16")[0]

            vae = WanVideoVAELoader()
            self.vae = vae.loadmodel(vae_name, precision="bf16")[0]

            slg = WanVideoSLG()
            self.slg = slg.process(blocks="2", start_percent=0.2, end_percent=0.7)[0]

            self.text_encode = WanVideoTextEncode()
            self.empty_embeds = WanVideoEmptyEmbeds()
            self.sampler = WanVideoSampler()
            self.decoder = WanVideoDecode()

    def inference(self, prompt, n_prompt=None, steps=6, num_frames=97, cfg=1.0, shift=2.0, width=832, height=480, seed=42):
        if n_prompt is None:
            # default n_prompt
            n_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards."
        with torch.no_grad():
            samples = self.sampler.process(
                model=self.wanvideo,
                text_embeds=self.text_encode.process(
                    t5=self.text_encoder,
                    model_to_offload=self.wanvideo,
                    positive_prompt=prompt,
                    negative_prompt=n_prompt,
                    force_offload=True,
                )[0],
                image_embeds=self.empty_embeds.process(num_frames=num_frames, width=width, height=height)[0],
                slg_args=self.slg,
                steps=steps,
                cfg=cfg,
                shift=shift,
                seed=seed,
                force_offload=True,
                scheduler="unipc",
                riflex_freq_index=0,
            )[0]
            images = self.decoder.decode(
                vae=self.vae, samples=samples, enable_vae_tiling=False, tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128
            )[0]
            output_video_path = save_video(images, seed=seed)
            return output_video_path


if __name__ == "__main__":
    with torch.no_grad():
        wanvideo = WanVideo()
        prompt = "A misty dawn unfolds in an ancient cherry-blossom grove as a sleek, silvery-furred cat in ornate samurai armor—with dark indigo and crimson plates, gold filigree, and cherry-blossom motifs—stands poised in a lone clearing surrounded by playful yet determined dog warriors in battered leather gear; in one fluid motion it draws a miniature katana (paw-shaped guard, red silk hilt) and wakizashi as its amber eyes lock onto its foes beneath a crimson headband, then darts forward in a graceful leap, low-tracking the camera alongside its lightning-fast parries and strikes, slow-motion flashes of steel catching pre-dawn light and scattering petals, the stylized paw-within-waves chest crest and swaying omamori charm hinting at its heritage, all underscored by soft wind, metallic clashes, distant growls, and a rising taiko drum swell, until it weaves between three attackers, halts center-frame, chest heaving with disciplined confidence, and the scene fades out on its silhouette against a cascade of blossoms."
        output_video_path = wanvideo.inference(prompt=prompt)
        print(f"Video saved to {output_video_path}")
