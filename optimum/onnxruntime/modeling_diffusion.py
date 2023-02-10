import logging
import os
import shutil
from abc import ABC
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from transformers.modeling_outputs import Seq2SeqLMOutput

import numpy as np

import onnxruntime as ort
from ..onnx.utils import _get_external_data_paths

from ..exporters.onnx import (
    export_models,
    get_stable_diffusion_models_for_export,
)
from ..exporters.tasks import TasksManager
from .modeling_ort import ORTModel
from .utils import (
    ONNX_WEIGHTS_NAME,
    get_provider_for_device,
    parse_device,
    validate_provider_availability,
    ORT_TO_NP_TYPE,
)
from .base import ORTModelPart

from huggingface_hub.utils import EntryNotFoundError

logger = logging.getLogger(__name__)


class ORTModelForStableDiffusion(ORTModel, ABC):
    """ """

    auto_model_class = StableDiffusionPipeline
    main_input_name = "input_ids"
    base_model_prefix = "onnx_model"

    def __init__(
        self,
        vae_decoder_session: ort.InferenceSession,
        text_encoder_session: ort.InferenceSession,
        # tokenizer: CLIPTokenizer,
        unet_session: ort.InferenceSession,
        # scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        # feature_extractor: CLIPFeatureExtractor,
        vae_encoder_session: Optional[ort.InferenceSession] = None,
        safety_checker: Optional[ort.InferenceSession] = None,
        # requires_safety_checker: bool = False,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        # preprocessors: Optional[List] = None,
        **kwargs,
    ):
        ABC.__init__(self)

        # preprocessors = [tokenizer, feature_extractor]
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
        self.config = None

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

    def _save_config(self, save_directory: Union[str, Path]):
        # self.config.save_pretrained(save_directory)
        # model.save_config(save_dir_path)
        pass

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

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
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
        model_path = Path(model_id)
        # Add infer_onnx_filename + _generate_regular_names_for_filename
        text_encoder_path = model_path / "text_encoder" / text_encoder_file_name
        unet_path = model_path / "unet" / unet_file_name
        vae_decoder_path = model_path / "vae_decoder" / vae_decoder_file_name

        preprocessors = None

        if model_path.is_dir():
            inference_sessions = cls.load_model(
                vae_decoder_path=vae_decoder_path,
                text_encoder_path=text_encoder_path,
                unet_path=unet_path,
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
            )
            new_model_save_dir = model_path
        else:
            attribute_name_to_filename = {
                "vae_decoder_model_name": vae_decoder_path.name,
                "text_encoder_model_name": text_encoder_path.name,
                "unet_model_name": unet_path.name,
            }
            paths = {}
            for attr_name, filename in attribute_name_to_filename.items():
                if filename is None:
                    continue
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=filename,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                # Download external data if present
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        subfolder=subfolder,
                        filename=filename + "_data",
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                except EntryNotFoundError:
                    # Model doesn't use external data
                    pass

                paths[attr_name] = Path(model_cache_path).name
            new_model_save_dir = Path(model_cache_path).parent

            inference_sessions = cls.load_model(
                vae_decoder_path=new_model_save_dir / paths["vae_decoder_model_name"],
                text_encoder_path=new_model_save_dir / paths["text_encoder_model_name"],
                unet_path=new_model_save_dir / paths["unet_model_name"],
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
            )

        # TODO : Remove model_save_dir
        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        if use_io_binding:
            logger.warning("OBinding is not yet available, `use_io_binding` set to False.")
            use_io_binding = False

        return cls(
            *inference_sessions,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
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
        use_cache: bool = True,
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
            use_cache=use_cache,
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
    def _load_config(cls, *args, **kwargs):
        # TODO : load config
        return None

class ORTModelTextEncoder(ORTModelPart):
    def forward(self, input_ids : np.ndarray):
        onnx_inputs = {
            "input_ids": input_ids,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs

class ORTModelUnet(ORTModelPart):
    def __init__(self, session: ort.InferenceSession, parent_model: "ORTModel"):
        super().__init__(session, parent_model)
        self.input_dtype = {inputs.name: ORT_TO_NP_TYPE[inputs.type] for inputs in self.session.get_inputs()}

    def forward(self, sample: np.ndarray, timestep : np.ndarray, encoder_hidden_states : np.ndarray):
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




