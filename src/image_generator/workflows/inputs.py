"""Named transient inputs, independent of HTTP uploads and GPU runtimes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InputSpec:
    name: str
    kind: str = "image"
    required: bool = True
    label: str = "Source image"
    help: str = "Choose a single-frame image. Source files are transient generation inputs."

    def parameter(self):
        from .parameters import Parameter

        return Parameter(
            self.name, self.label, "file", "Source image", scope="inputs", required=self.required, help=self.help
        )


def validate_inputs(specs, inputs):
    declared = {spec.name: spec for spec in specs}
    unknown = inputs.keys() - declared.keys()
    if unknown:
        raise ValueError(f"Unexpected inputs: {', '.join(sorted(unknown))}")
    for name, spec in declared.items():
        path = inputs.get(name)
        if path is None:
            if spec.required:
                raise ValueError(f"{name} is required")
        elif not path.is_file():
            raise ValueError(f"{name} must be an existing local file")
