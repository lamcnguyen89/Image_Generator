"""Reusable named-adapter lifecycle for standard Diffusers pipelines."""


class DiffusersAdapters:
    def _load_lora(self, selection, **options):
        path = selection.asset.asset_paths["weights"]
        self._pipeline.load_lora_weights(
            str(path.parent),
            weight_name=path.name,
            adapter_name=selection.adapter_name,
            local_files_only=True,
            **options,
        )

    def _configure_loras(self, request, *, restore_overwritten=False):
        self.report_progress("resetting_loras")
        if self._has_adapters:
            options = {"reset_to_overwritten_params": True} if restore_overwritten else {}
            self._pipeline.unload_lora_weights(**options)
            self._has_adapters = False
        self.report_progress("loading_loras", 0, len(request.loras))
        for index, selection in enumerate(request.loras):
            self._has_adapters = True
            options = {}
            try:
                self._load_lora(selection, **options)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load LoRA {selection.asset.id!r}; check model and loader compatibility: {exc}"
                ) from exc
            self.report_progress("loading_loras", index + 1, len(request.loras))
        if request.loras:
            self._pipeline.set_adapters(
                [item.adapter_name for item in request.loras],
                adapter_weights=[float(item.strength) for item in request.loras],
            )
        self.report_progress("loras_ready", len(request.loras), len(request.loras))
