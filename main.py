import os
import io
import base64
import argparse
import json
import requests
from tempfile import NamedTemporaryFile, TemporaryDirectory
from secrets import compare_digest
from typing import List, Optional, Type, Dict, Any
from tqdm import tqdm
from collections import namedtuple
from PIL import Image
from llava.utils import disable_torch_init

# set up server with uvicorn
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

disable_torch_init()
import torch
from torch import dtype as TorchDtype
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from transformers import TextStreamer

PREPROCESSOR = {}
MODEL_CONFIG = {}  # device, dtype
LOADED_STATE = {}  # model, tokenizer, image_processor, context_len, roles, conv
TEMP_IMAGE_DIR = TemporaryDirectory()  # temporary directory for images
TEMP_IMAGE_CACHE = {}  # temporary image cache


def cast_to_device(tensor):
    """
    Cast the tensor to the device.
    """
    return tensor.to(MODEL_CONFIG["device"], dtype=MODEL_CONFIG["dtype"])


class LLavaArguments(
    namedtuple(
        "LLavaArguments",
        [
            "model_path",
            "model_base",
            "device",
            "cuda_devices",
            "dtype",
            "model",
            "image_processor",
            "tokenizer",
            "conv_mode",
            "load_8bit",
            "load_4bit",
        ],
        defaults=[
            "Lin-Chen/ShareGPT4V-7B",
            None,
            "cuda",
            None,
            torch.bfloat16,
            None,
            None,
            None,
            None,
            False,
            False,
        ],
    )
):
    """
    Arguments for LLaVA.
    """

    model_path: str = "Lin-Chen/ShareGPT4V-7B"
    model_base: Optional[str] = None
    device: str = "cuda"
    cuda_devices: Optional[str] = None
    dtype: TorchDtype = torch.bfloat16
    model: Optional[Type] = None
    image_processor: Optional[Type] = None
    tokenizer: Optional[Type] = None
    conv_mode: Optional[str] = None
    load_8bit: bool = False
    load_4bit: bool = False

    def __init__(self, **kwargs):
        """
        Initialize the LLaVA arguments.
        """
        # setattr for fields, filtered by namedtuple._fields
        for k, v in kwargs.items():
            if k in self._fields:
                setattr(self, k, v)

    @staticmethod
    def configure_argparser() -> argparse.ArgumentParser:
        """
        Configure the argument parser for LLaVA.
        """
        parser = argparse.ArgumentParser(
            description="LLavaPipeline",
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default="Lin-Chen/ShareGPT4V-7B",
            help="path to the model",
        )
        parser.add_argument(
            "--model-base",
            type=str,
            default=None,
            help="base model for the model",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="device to use for inference",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="bfloat16",
            help="dtype to use for inference",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="model to use for inference",
        )
        parser.add_argument(
            "--image-processor",
            type=str,
            default=None,
            help="image processor to use for inference",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=None,
            help="tokenizer to use for inference",
        )
        parser.add_argument(
            "--conv-mode",
            type=str,
            default=None,
            help="conversation mode to use for inference",
        )
        parser.add_argument(
            "--load-8bit",
            action="store_true",
            help="load 8bit model",
        )
        parser.add_argument(
            "--load-4bit",
            action="store_true",
            help="load 4bit model",
        )
        return parser

    @staticmethod
    def parse_arguments(parser: argparse.ArgumentParser) -> "LLavaArguments":
        """
        Parse the arguments for LLaVA.
        """
        args = parser.parse_args()
        kwargs_to_accept = set(LLavaArguments._fields)
        # parse dtype from string
        if args.dtype:
            # bf16 / bfloat16
            if args.dtype.lower() in ["bf16", "bfloat16"]:
                args.dtype = torch.bfloat16
            # fp32 / float32
            elif args.dtype.lower() in ["fp32", "float32", "float"]:
                args.dtype = torch.float32
            # fp16 / float16
            elif args.dtype.lower() in ["fp16", "float16", "half"]:
                args.dtype = torch.float16
            elif args.dtype.lower() in ["int8", "int"]:
                args.dtype = torch.int8
            elif args.dtype.lower() in ["int16", "short"]:
                args.dtype = torch.int16
            elif args.dtype.lower() in ["int32", "long"]:
                args.dtype = torch.int32
            else:
                raise ValueError(f"Unknown dtype {args.dtype}")
        return LLavaArguments(
            **{k: v for k, v in vars(args).items() if k in kwargs_to_accept}
        )


class ServerArguments(dict):
    """
    Handles server arguments for this session.
    """

    kwargs_to_handle = [
        "port",
        "api_auth_file",
        "api_auth_pair",
        "launch_option",
        "test_inference",
        "test_api",
    ]
    port: int = 8000
    api_auth_file: Optional[str] = "api_auth.json"
    api_auth_pair: Optional[str] = "master:password"  # username:password
    launch_option: Optional[str] = "gradio"  # uvicorn, gradio(share)
    test_inference: bool = False  # test inference
    test_api: bool = False  # test api, exclusive with test_inference

    def __init__(self, **kwargs):
        """
        Initialize the server arguments.
        """
        # filter out kwargs that are not in the class
        super().__init__(
            **{k: v for k, v in kwargs.items() if k in ServerArguments.kwargs_to_handle}
        )
        # setattr for fields
        for k, v in self.items():
            setattr(self, k, v)

    @staticmethod
    def attach_arguments(parser: argparse.ArgumentParser) -> None:
        """
        Attach the arguments for the server.
        """
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="port to use for the server",
        )
        parser.add_argument(
            "--api-auth-file",
            type=str,
            default="api_auth.json",
            help="path to the api auth file",
        )
        parser.add_argument(
            "--api-auth-pair",
            type=str,
            default="master:password",
            help="api auth pair",
        )
        parser.add_argument(
            "--launch-option",
            type=str,
            default="uvicorn",
            help="launch option, can be uvicorn or gradio(to open gradio interface into public))",
        )
        parser.add_argument(
            "--test-inference", action="store_true", help="Test inference"
        )
        parser.add_argument(
            "--test-api",
            action="store_true",
            help="Test api, exclusive with test-inference",
        )

    @staticmethod
    def get_auth_pairs(args: "ServerArguments") -> Dict[str, str]:
        """
        Get the auth pairs.
        """
        # get the auth pairs
        auth_pairs = {}
        if args.api_auth_file and os.path.isfile(args.api_auth_file):
            with open(args.api_auth_file, "r", encoding="utf-8") as f:
                auth_pairs = json.load(f)
        if args.api_auth_pair:
            username, password = args.api_auth_pair.split(":")
            auth_pairs[username] = password
        return auth_pairs

    @staticmethod
    def parse_arguments(parser: argparse.ArgumentParser) -> "ServerArguments":
        """
        Parse the arguments for the server.
        """
        args = parser.parse_args()
        kwargs_to_accept = set(ServerArguments.kwargs_to_handle)
        return ServerArguments(
            **{k: v for k, v in vars(args).items() if k in kwargs_to_accept}
        )


class InferenceArguments(dict):
    """
    Handles inference arguments for this session.
        @param num_samples: number of samples to generate
        @param inference_kwargs: kwargs for inference
        @param delimeter: delimeter for the prompt
        @param text_or_path: text or path to the text
        @param image_or_path: image or path to the image

    """

    kwargs_to_handle = [
        "num_samples",
        "inference_kwargs",
        "delimeter",
        "text_or_path",
        "image_or_path",
        "prompt_format",
    ]
    num_samples = 5
    inference_kwargs = {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "use_cache": True,
        "do_sample": True,  # this should be True if temperature > 0, non-deterministic
    }
    delimeter = ". "
    text_or_path = ""
    image_or_path = ""
    prompt_format = ""

    def __init__(self, **kwargs):
        """
        Initialize the inference arguments.
        """
        # filter out kwargs that are not in the class
        super().__init__(
            **{
                k: v
                for k, v in kwargs.items()
                if k in InferenceArguments.kwargs_to_handle
            }
        )
        # setattr for fields
        for k, v in self.items():
            setattr(self, k, v)

    def get_prompt_format(self):
        """
        Get the prompt format.
        """
        if self.prompt_format:
            # from file or string
            if os.path.isfile(self.prompt_format):
                with open(self.prompt_format, "r", encoding="utf-8") as f:
                    return f.read()
            return self.prompt_format
        else:
            return PROMPT_DEFAULT


class InferenceArgumentsInput(BaseModel):
    """
    Handles inference arguments for this session.
    """

    num_samples: int = 5
    inference_kwargs: Dict[str, Any] = {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "use_cache": True,
        "do_sample": True,  # this should be True if temperature > 0, non-deterministic
    }
    delimeter: str = ". "
    text_or_path: str = ""
    image_or_path: str = ""


class InferenceArgumentsOutput(BaseModel):
    """
    Returns the sampled text.
    """

    samples_texts: List[str] = []


def register_preprocessor(preprocessor, preprocessor_config):
    """
    Register preprocessor.
    """
    PREPROCESSOR["preprocessor"] = preprocessor
    PREPROCESSOR["preprocessor_config"] = preprocessor_config


PROMPT_DEFAULT = """SYSTEM: A chat between a curious human and an artificial intelligence assistant.

The assistant gives helpful, detailed, and polite answers to the human's questions.

Below is an instruction that describes a task. Write a response that appropriately completes the request.

<instruction>

- Avoid describing in numerical order.

- Maintain a neutral tone in the description to the greatest extent possible.

<\instruction>

<reference>

{original_tags}

<\reference>

USER: Please adhere to the following instructions in your response.

1. The attached comma-separated tags form part of the pre-checked reference for the image.

2. Please refrain from directly using the provided reference in your description, as it is not explicitly depicted in the image.

3. Meticulously analyze the following image details and furnish a comprehensive description.

4. Limit your description to elements present in the given image, refrain from providing details about elements that are not visually represented.

<image></s>

ASSISTANT: """


def format_prompt(original_tag: str, prompt_template: str):
    """
    Format the prompt.
    """
    prompt_template = handle_text_web(prompt_template)
    # find {original_tags}, if not found, use order format
    return prompt_template.format(original_tags=original_tag)


def handle_text_web(text_or_path: str) -> str:
    """
    Handles download and return of text.
    """
    if text_or_path.startswith("http"):
        print(f"Downloading text from {text_or_path}")
        # download the text
        text_file = requests.get(text_or_path)
        # check response
        if text_file.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text download failed with status code {text_file.status_code}",
            )
        text_file_content = text_file.content
        # return the text, we assume it is utf-8 encoded
        return text_file_content.decode("utf-8")
    else:
        return text_or_path


def handle_temp_image(image_path_or_str: str) -> str:
    """
    Handles download and return of temporary image.
    """
    if image_path_or_str in TEMP_IMAGE_CACHE:
        return TEMP_IMAGE_CACHE[image_path_or_str]
    if image_path_or_str.startswith("http"):
        print(f"Downloading image from {image_path_or_str}")
        # download the image
        image_file = requests.get(image_path_or_str)
        # check response
        if image_file.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image download failed with status code {image_file.status_code}",
            )
        image_file_content = image_file.content
        # get the image extension from header
        file_extension = image_file.headers["Content-Type"].split("/")[-1]
        # save the image to temporary directory
        with NamedTemporaryFile(
            dir=TEMP_IMAGE_DIR.name, delete=False, suffix=f".{file_extension}"
        ) as f:
            f.write(image_file_content)
            image_file_path = f.name
        TEMP_IMAGE_CACHE[image_path_or_str] = image_file_path
        return image_file_path
    else:
        return image_path_or_str


def load_image(image_path_or_str: str) -> Image.Image:
    """
    Handles loading the image.
    Image can be one of three type:
        - path to image
        - base64 encoded image
        - Web Address for raw image
    """
    image_path_or_str = handle_temp_image(image_path_or_str)
    if os.path.isfile(image_path_or_str):
        image_file = image_path_or_str
    else:
        # base64 encoded image
        image_file = io.BytesIO(base64.b64decode(image_path_or_str))
    image = Image.open(image_file)
    # handle RGBA images, transparent part should be white
    if image.mode == "RGBA":
        print("RGBA image detected, converting to RGB")
        # prepare a white background image
        bg = Image.new("RGB", image.size, (255, 255, 255))
        # composite the images
        bg.paste(image, mask=image.split()[3])
        image = bg
    else:
        image = image.convert("RGB")
    return image


def clean_up_temp_image() -> None:
    """
    Clean up the temporary directory.
    """
    TEMP_IMAGE_DIR.cleanup()
    TEMP_IMAGE_CACHE.clear()


def get_image_tensor(image_str) -> torch.Tensor:
    """
    Get the image tensor.
    Processes required instructions on the image.
    """
    image = load_image(image_str)
    image_tensor = process_images(
        [image], PREPROCESSOR["preprocessor"], PREPROCESSOR["preprocessor_config"]
    )
    if isinstance(
        image_tensor, list
    ):  # preprocessor result was list, wrap to tensor with device
        image_tensor = [cast_to_device(image.to) for image in image_tensor]
    else:
        image_tensor = cast_to_device(image_tensor)
    return image_tensor


def load_llava(args: LLavaArguments):
    """
    Load the LLaVA model.
    """
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )
    MODEL_CONFIG["device"] = args.device
    MODEL_CONFIG["dtype"] = args.dtype
    register_preprocessor(image_processor, model.config)

    if args.load_8bit or args.load_4bit:
        if args.load_4bit:
            print(f"{model_name} loaded in 4bit")

        elif args.load_8bit:
            print(f"{model_name} loaded in 8bit")
    # get suggested conversation mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "gpt4" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
        conv_mode = args.conv_mode
    print(f"conversation model : {conv_mode}")
    conv = conv_templates[conv_mode].copy()

    if "mpt" in model_name.lower():
        roles = ("user", "assistant")

    else:
        roles = conv.roles
    model.to(device=args.device, dtype=args.dtype)
    return model, tokenizer, image_processor, context_len, roles, conv


def inference(args: InferenceArguments, model, tokenizer, conv) -> List[str]:
    """
    Inference with given arguments.
    """
    image_tensor = get_image_tensor(args.image_or_path)
    text_formatted = format_prompt(args.text_or_path, args.get_prompt_format())

    if conv.sep_style == SeparatorStyle.TWO:
        stop_str = conv.sep2
    else:
        stop_str = conv.sep

    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs_container = []
    with torch.inference_mode():
        for _ in tqdm(range(args.num_samples)):
            eos_end = False
            input_ids = (
                tokenizer_image_token(
                    text_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(model.device)
            )
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                **args.inference_kwargs,
                streamer=streamer,
                stopping_criteria=[stopping_criteria],
            )
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
            if outputs.endswith("</s>"):
                eos_end = True
            if eos_end and not outputs.endswith("</s>"):
                outputs = outputs + "</s>"
            outputs_container.append(outputs)
    clean_up_temp_image()
    return outputs_container


def auth(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    """
    Handles authentication.
    """
    auth_pairs = ServerArguments.get_auth_pairs(ServerArguments)
    if not auth_pairs:
        return True
    if credentials.username in auth_pairs:
        if compare_digest(credentials.password, auth_pairs[credentials.username]):
            return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )


def bind_inference_api(app: FastAPI):
    """
    Binds inference api.
    """
    router = APIRouter()

    @app.post(
        "/inference",
        response_model=InferenceArgumentsOutput,
        dependencies=[Depends(auth)],
    )
    def wrap_inference(args: InferenceArguments) -> InferenceArgumentsOutput:
        """
        Wrap inference.
        """
        # check if model is loaded
        assert LOADED_STATE["model"] is not None, "Model is not loaded"
        outputs = inference(
            args, LOADED_STATE["model"], LOADED_STATE["tokenizer"], LOADED_STATE["conv"]
        )
        return InferenceArgumentsOutput(samples_texts=outputs)

    app.include_router(router)
    app.add_api_route(
        "/inference",
        endpoint=wrap_inference,
        methods=["POST"],
        response_model=InferenceArgumentsOutput,
        dependencies=[Depends(auth)],
    )


def test_inference():
    """
    Test inference.
    """
    # test inference
    test_args = InferenceArguments(
        num_samples=5,
        inference_kwargs={
            "temperature": 0.2,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "use_cache": True,
            "do_sample": True,  # this should be True if temperature > 0, non-deterministic
        },
        delimeter=". ",
        text_or_path="1girl, white hair, short hair, lightblue eyes, flowers, light, sitting",  # tags
        image_or_path="https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/f6929d4d-5991-4c10-b013-0743ffc8e207",  # image
    )
    result = inference(
        test_args,
        LOADED_STATE["model"],
        LOADED_STATE["tokenizer"],
        LOADED_STATE["conv"],
    )
    print(result)


def test_api(url_base: str, port: int, auth_pair: str):
    """
    Test inference with api.
    """
    import time

    time.sleep(5)
    if url_base.endswith("/"):
        url_base = url_base[:-1]
    endpoint = f"{url_base}:{port}/inference"
    test_args = InferenceArgumentsInput(
        num_samples=5,
        inference_kwargs={
            "temperature": 0.2,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "use_cache": True,
            "do_sample": True,  # this should be True if temperature > 0, non-deterministic
        },
        delimeter=". ",
        text_or_path="1girl, white hair, short hair, lightblue eyes, flowers, light, sitting",  # tags
        image_or_path="https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/f6929d4d-5991-4c10-b013-0743ffc8e207",
    )
    session = requests.Session()
    session.auth = tuple(auth_pair.split(":"))
    result = session.post(endpoint, json=test_args.dict())
    if result.status_code != 200:
        print(f"Test api failed with status code {result.status_code}, {result.text}")
    else:
        print("Test api passed")


def gradio_run(port):
    """
    Run gradio interface.
    """
    try:
        import gradio as gr  # pylint: disable=import-outside-toplevel
    except ImportError:
        # gradio not installed
        print("Gradio not installed, skipping gradio interface")
        return FastAPI()
    # simple title
    with gr.Blocks() as interface:
        gr.Markdown("## LLaVA")
    gradio_app, local_url, share_url = interface.launch(
        share=True,
        server_port=port,
        app_kwargs={
            "docs_url": "/docs",
            "redoc_url": "/redoc",
        },
        prevent_thread_lock=True,
    )
    return gradio_app, local_url


def main():
    """
    Set up model
    """
    parser = LLavaArguments.configure_argparser()
    # attach server arguments
    ServerArguments.attach_arguments(parser)
    args = LLavaArguments.parse_arguments(parser)
    server_args = ServerArguments.parse_arguments(parser)
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    model, tokenizer, _, _, _, conv = load_llava(args)
    LOADED_STATE["model"] = model
    LOADED_STATE["tokenizer"] = tokenizer
    LOADED_STATE["conv"] = conv
    print(f"Server running on port {server_args.port}")
    if not server_args.test_api and server_args.test_inference:
        test_inference()
    # run the server
    # run gradio frontend with share option if specified
    if server_args.launch_option == "gradio":
        app, local_url = gradio_run(server_args.port)
    else:
        app, local_url = FastAPI(), "http://127.0.0.1"
    bind_inference_api(app)
    if server_args.test_api:
        import threading

        # run test api in another thread
        threading.Thread(
            target=test_api,
            args=(local_url, server_args.port, server_args.api_auth_pair),
        ).start()
    uvicorn.run(app, port=server_args.port)


if __name__ == "__main__":
    main()
