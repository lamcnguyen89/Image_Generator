from image_generator.generation import request_snapshot
from image_generator.workflows.base import GeneratedBatch, Workflow


class DemoRuntime(Workflow):
    def __init__(self, **options):
        self.options = options

    def generate(self, request):
        from PIL import Image

        self.validate(request)
        self.report_progress("denoising", 1, 1)
        seed = request.settings["seed"] or 42
        return GeneratedBatch((Image.new("RGB", (8, 8), "purple"),), {**request_snapshot(request), "seeds": [seed]})

    def close(self):
        pass
