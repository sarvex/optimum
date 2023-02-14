import importlib
import inspect
import logging
import os
import shutil
from abc import ABC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import ConfigMixin, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer

import onnxruntime as ort

from ..exporters.onnx import (
    export_models,
    get_stable_diffusion_models_for_export,
)
from ..exporters.tasks import TasksManager
from ..onnx.utils import _get_external_data_paths
from .base import ORTModelPart
from .modeling_ort import ORTModel
from .utils import (
    ONNX_WEIGHTS_NAME,
    ORT_TO_NP_TYPE,
    get_provider_for_device,
    parse_device,
    validate_provider_availability,
)


logger = logging.getLogger(__name__)


class ORTModelForStableDiffusion(ORTModel, ConfigMixin, ABC):
    """ """

    auto_model_class = StableDiffusionPipeline
    main_input_name = "input_ids"
    base_model_prefix = "onnx_model"
    config_name = "model_index.json"

    def __init__(
        self,
        vae_decoder_session: ort.InferenceSession,
        text_encoder_session: ort.InferenceSession,
        unet_session: ort.InferenceSession,
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: CLIPFeatureExtractor,
        config: Dict[str, Any],
        safety_checker_session: Optional[ort.InferenceSession] = None,
        # requires_safety_checker: bool = False,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        ABC.__init__(self)

        self.shared_attributes_init(
            vae_decoder_session,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=None,
        )
        self.vae_decoder = ORTModelVaeDecoder(vae_decoder_session, self)
        self.vae_decoder_model_path = Path(vae_decoder_session._model_path)
        self.text_encoder = ORTModelTextEncoder(text_encoder_session, self)
        self.text_encoder_model_path = Path(text_encoder_session._model_path)
        self.unet = ORTModelUnet(unet_session, self)
        self.unet_model_path = Path(unet_session._model_path)
        self._config = config
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None
        self._internal_dict = config

    @staticmethod
    def load_model(
        vae_decoder_path: Optional[Union[str, Path]],
        text_encoder_path: Union[str, Path],
        unet_path: Union[str, Path],
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict] = None,
    ):
        vae_decoder_session = ORTModel.load_model(vae_decoder_path, provider, session_options, provider_options)
        text_encoder_session = ORTModel.load_model(text_encoder_path, provider, session_options, provider_options)
        unet_session = ORTModel.load_model(unet_path, provider, session_options, provider_options)

        return vae_decoder_session, text_encoder_session, unet_session

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        text_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        unet_file_name: str = ONNX_WEIGHTS_NAME,
        vae_decoder_file_name: str = ONNX_WEIGHTS_NAME,
        **kwargs,
    ):
        save_directory = Path(save_directory)
        src_to_dst_path = {
            self.text_encoder_model_path: save_directory / "text_encoder" / text_encoder_file_name,
            self.unet_model_path: save_directory / "unet" / unet_file_name,
            self.vae_decoder_model_path: save_directory / "vae_decoder" / vae_decoder_file_name,
        }

        # Add external data paths in case of large models
        # TODO: Modify _get_external_data_paths to give dictionnary
        src_paths = list(src_to_dst_path.keys())
        dst_paths = list(src_to_dst_path.values())
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)

        self.save_config(save_directory)
        self.tokenizer.save_pretrained(save_directory.joinpath("tokenizer"))
        self.scheduler.save_pretrained(save_directory.joinpath("scheduler"))
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory.joinpath("feature_extractor"))

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        text_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        unet_file_name: str = ONNX_WEIGHTS_NAME,
        vae_decoder_file_name: str = ONNX_WEIGHTS_NAME,
        subfolder: str = "",
        local_files_only: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        model_id = str(model_id)
        init_dict, unused_kwargs, _ = cls.extract_init_dict(config)
        sub_model_to_load = set(init_dict.keys()).intersection({"feature_extractor", "tokenizer", "scheduler"})
        sub_models = {}

        if not os.path.isdir(model_id):
            allow_patterns = [os.path.join(k, "*") for k in config.keys() if not k.startswith("_")]
            allow_patterns += list(
                {
                    text_encoder_file_name,
                    unet_file_name,
                    vae_decoder_file_name,
                    SCHEDULER_CONFIG_NAME,
                    CONFIG_NAME,
                    cls.config_name,
                }
            )
            # Download all allow_patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin"],
            )
        new_model_save_dir = Path(model_id)

        for name in sub_model_to_load:
            library = importlib.import_module(init_dict[name][0])
            class_obj = getattr(library, init_dict[name][1])
            load_method = getattr(class_obj, "from_pretrained")
            # Check if the module is in a subdirectory
            if (new_model_save_dir / name).is_dir():
                sub_models[name] = load_method(new_model_save_dir / name)
            else:
                sub_models[name] = load_method(new_model_save_dir)

        # Add infer_onnx_filename + _generate_regular_names_for_filename
        text_encoder_path = new_model_save_dir / "text_encoder" / text_encoder_file_name
        unet_path = new_model_save_dir / "unet" / unet_file_name
        vae_decoder_path = new_model_save_dir / "vae_decoder" / vae_decoder_file_name

        inference_sessions = cls.load_model(
            vae_decoder_path=vae_decoder_path,
            text_encoder_path=text_encoder_path,
            unet_path=unet_path,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
        )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        if use_io_binding:
            logger.warning("OBinding is not yet available, `use_io_binding` set to False.")
            use_io_binding = False

        return cls(
            *inference_sessions,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            tokenizer=sub_models["tokenizer"],
            scheduler=sub_models["scheduler"],
            feature_extractor=sub_models.pop("feature_extractor", None),
            config=config,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "ORTModelForStableDiffusion":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        model = TasksManager.get_model_from_task(
            task,
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        output_names = [
            os.path.join("text_encoder", ONNX_WEIGHTS_NAME),
            os.path.join("unet", ONNX_WEIGHTS_NAME),
            os.path.join("vae_encoder", ONNX_WEIGHTS_NAME),
            os.path.join("vae_decoder", ONNX_WEIGHTS_NAME),
        ]
        models_and_onnx_configs = get_stable_diffusion_models_for_export(model)

        model.save_config(save_dir_path)

        # maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)
        model.tokenizer.save_pretrained(save_dir_path.joinpath("tokenizer"))
        model.scheduler.save_pretrained(save_dir_path.joinpath("scheduler"))
        model.feature_extractor.save_pretrained(save_dir_path.joinpath("feature_extractor"))

        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=save_dir_path,
            output_names=output_names,
        )
        return cls._from_pretrained(
            save_dir_path,
            config=config,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            use_io_binding=use_io_binding,
            model_save_dir=save_dir,
        )

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`torch.device` or `str` or `int`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTModel`: the model placed on the requested device.
        """
        device, provider_options = parse_device(device)
        provider = get_provider_for_device(device)
        validate_provider_availability(provider)  # raise error if the provider is not available
        self.device = device
        self.text_encoder.session.set_providers([provider], provider_options=[provider_options])
        self.unet.session.set_providers([provider], provider_options=[provider_options])
        self.vae_decoder.session.set_providers([provider], provider_options=[provider_options])
        self.providers = self.text_encoder.session.get_providers()
        return self

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    # Copied from https://github.com/huggingface/diffusers/blob/v0.12.1/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L115
    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

        if not np.array_equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]
        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Adapted from https://github.com/huggingface/diffusers/blob/v0.12.1/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L192
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if generator is None:
            generator = np.random

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
        )

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="np"
            ).pixel_values.astype(image.dtype)

            image, has_nsfw_concepts = self.safety_checker(clip_input=safety_checker_input, images=image)

            # There will throw an error if use safety_checker batchsize>1
            images, has_nsfw_concept = [], []
            for i in range(image.shape[0]):
                image_i, has_nsfw_concept_i = self.safety_checker(
                    clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                )
                images.append(image_i)
                has_nsfw_concept.append(has_nsfw_concept_i[0])
            image = np.concatenate(images)
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


class ORTModelTextEncoder(ORTModelPart):
    def forward(self, input_ids: np.ndarray):
        onnx_inputs = {
            "input_ids": input_ids,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs


class ORTModelUnet(ORTModelPart):
    def __init__(self, session: ort.InferenceSession, parent_model: "ORTModel"):
        super().__init__(session, parent_model)
        self.input_dtype = {inputs.name: ORT_TO_NP_TYPE[inputs.type] for inputs in self.session.get_inputs()}

    def forward(self, sample: np.ndarray, timestep: np.ndarray, encoder_hidden_states: np.ndarray):
        onnx_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs


class ORTModelVaeDecoder(ORTModelPart):
    def forward(self, latent_sample: np.ndarray):
        onnx_inputs = {
            "latent_sample": latent_sample,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs
