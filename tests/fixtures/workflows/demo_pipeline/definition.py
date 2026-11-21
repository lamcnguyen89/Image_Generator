from pathlib import Path

from image_generator.workflows.definition import ModeDefinition, WorkflowDefinition
from image_generator.workflows.inputs import InputSpec
from image_generator.workflows.parameters import Parameter
from image_generator.workflows.runtime import RuntimeOption
from image_generator.workflows.ui import UIDefinition

DEFAULTS = {
    "prompt": "",
    "negative_prompt": "",
    "seed": 42,
    "temperature": 2,
    "num_images_per_prompt": 1,
    "use_filter": True,
    "loras": [],
}


def parameters(model, mode):
    return UIDefinition(
        model.defaults["workflow"],
        mode,
        (mode,),
        (
            Parameter("prompt", "Prompt", "textarea", "Prompts", required=True),
            Parameter("negative_prompt", "Negative prompt", "textarea", "Prompts"),
            Parameter(
                "temperature", "Custom temperature", "integer", "Sampling", minimum=1, maximum=5, cli_flags=("--heat",)
            ),
            Parameter("seed", "Seed", "integer", "Output", nullable=True, minimum=0),
            Parameter("num_images_per_prompt", "Image count", "integer", "Output", minimum=1),
            Parameter("use_filter", "Use filter", "boolean", "Advanced"),
        ),
    )


def validate_settings(model, mode, settings):
    if settings["temperature"] == 5 and not settings["use_filter"]:
        raise ValueError("High temperature requires the filter")


def validate_request(request):
    if request.settings["temperature"] == 4:
        raise ValueError("Synthetic asset validation failure")


DEFINITION = WorkflowDefinition(
    name="demo_pipeline",
    family="sdxl",
    runtime=("demo_pipeline.runtime", "DemoRuntime"),
    defaults=DEFAULTS,
    modes=(ModeDefinition("reference_image", inputs=(InputSpec("input_image"), InputSpec("mask_image"))),),
    parameters=parameters,
    validate_settings=validate_settings,
    validate_request=validate_request,
    default_model_id="demo-model",
    asset_root=Path(__file__).parent,
    scripts=("controls.js",),
    styles=("controls.css",),
    group_partials=(("Sampling", "controls/demo.html"),),
    extra_runtime_options=(RuntimeOption("demo_device", str, "Test backend device"),),
)
